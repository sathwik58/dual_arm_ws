"""Window to choose a shape, enter its size (with safe limits) and send it to the Panda.

  ros2 run pattern_drawer draw_gui

Left : list of shapes.
Right: the selected shape's values. Every value has a locked range; the maximum is a
       safety fraction of what still fits inside the Panda's reachable area, so the arm
       is never asked for a shape it cannot draw.
Buttons: Random (good random values), Draw (runs draw_node; optionally clears the old ink
         first and scans + recognises + reports after), Clear (removes all ink), Stop.
         The terminal output shows in the log box.
"""
import os
import queue
import random
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

import yaml

WORKSPACE = os.path.expanduser('~/panda_workspace.yaml')
DEFAULT_AREA = {'x_min': 0.10, 'x_max': 0.55, 'y_min': -0.50, 'y_max': 0.50}  # last map
EDGE = 0.03          # keep every line at least 3 cm inside the reachable area
SAFETY = 0.85        # max size = 85 % of what would just fit
MIN_R = 0.03         # smaller than this the camera / CNN cannot read it well
SIM = ['--ros-args', '-p', 'use_sim_time:=true']
# 'timeout 20' = give up after 20 s if nobody offers the service (pen_trail not running)
CLEAR = ['timeout', '20', 'ros2', 'service', 'call', '/panda/clear_ink', 'std_srvs/srv/Trigger', '{}']

# shape -> which values it needs (draw_node parameter names)
SHAPES = {
    'square_circle': ['center_x', 'center_y', 'square_size', 'circle_radius'],
    'circle':        ['center_x', 'center_y', 'circle_radius'],
    'triangle':      ['center_x', 'center_y', 'circle_radius'],
    'polygon':       ['center_x', 'center_y', 'circle_radius', 'sides'],
    'star':          ['center_x', 'center_y', 'circle_radius'],
    'spiral':        ['center_x', 'center_y', 'circle_radius'],
    'center_square': [],      # size comes from workspace_map
    'max_square':    [],
    'rectangle':     [],
}
LABELS = {'center_x': 'Centre X (m)', 'center_y': 'Centre Y (m)',
          'square_size': 'Square side (m)', 'circle_radius': 'Radius (m)', 'sides': 'Sides'}


def safe_area():
    """Reachable rectangle from workspace_map, shrunk by EDGE on every side."""
    try:
        with open(WORKSPACE) as f:
            a = (yaml.safe_load(f) or {})['rectangle']
    except (OSError, KeyError, TypeError):
        a = DEFAULT_AREA
    return (a['x_min'] + EDGE, a['x_max'] - EDGE, a['y_min'] + EDGE, a['y_max'] - EDGE)


class DrawGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Panda drawing')
        self.area = safe_area()
        self.vars = {k: tk.StringVar() for k in LABELS}
        self.limit_labels = {}
        self.proc = None
        self.busy = False
        self.log_q = queue.Queue()

        # ---- left: shape list
        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky='ns')
        ttk.Label(left, text='Shape').pack(anchor='w')
        self.shape_list = tk.Listbox(left, height=len(SHAPES), exportselection=False)
        for s in SHAPES:
            self.shape_list.insert('end', s)
        self.shape_list.pack(fill='y', expand=True)
        self.shape_list.bind('<<ListboxSelect>>', lambda e: self.show_fields())

        # ---- right: values, limits, buttons
        self.right = ttk.Frame(self, padding=8)
        self.right.grid(row=0, column=1, sticky='nsew')
        x0, x1, y0, y1 = self.area
        ttk.Label(self.right, text=f'Safe area  x {x0:.2f}..{x1:.2f}   y {y0:.2f}..{y1:.2f}',
                  foreground='gray').grid(row=0, column=0, columnspan=3, sticky='w')
        self.fields = ttk.Frame(self.right)
        self.fields.grid(row=1, column=0, columnspan=3, sticky='w', pady=6)

        self.clear_first = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.right, text='Clear old ink before drawing',
                        variable=self.clear_first).grid(row=2, column=0, columnspan=4, sticky='w')
        self.full_run = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.right, text='Then scan + recognise + report',
                        variable=self.full_run).grid(row=3, column=0, columnspan=4, sticky='w')
        ttk.Button(self.right, text='Random', command=self.randomize).grid(row=4, column=0, pady=6)
        self.draw_btn = ttk.Button(self.right, text='Draw', command=self.draw)
        self.draw_btn.grid(row=4, column=1)
        self.clear_btn = ttk.Button(self.right, text='Clear', command=self.clear)
        self.clear_btn.grid(row=4, column=2)
        ttk.Button(self.right, text='Stop', command=self.stop).grid(row=4, column=3)
        self.status = ttk.Label(self.right, text='Select a shape', foreground='blue')
        self.status.grid(row=5, column=0, columnspan=4, sticky='w')

        # ---- bottom: log
        self.log = tk.Text(self, height=14, width=90, state='disabled')
        self.log.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=8, pady=8)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        for v in self.vars.values():                 # re-check limits on every key press
            v.trace_add('write', lambda *a: self.check())
        self.shape_list.selection_set(0)
        self.show_fields()
        self.randomize()
        self.after(100, self.pump_log)

    # ------------------------------------------------------------ fields
    def shape(self):
        sel = self.shape_list.curselection()
        return self.shape_list.get(sel[0]) if sel else None

    def show_fields(self):
        for w in self.fields.winfo_children():
            w.destroy()
        self.limit_labels = {}
        if not self.busy:
            self.status.configure(text='Ready', foreground='blue')
        names = SHAPES[self.shape()]
        if not names:
            ttk.Label(self.fields, text='Size comes from workspace_map - nothing to enter.'
                      ).grid(row=0, column=0)
        for i, name in enumerate(names):
            ttk.Label(self.fields, text=LABELS[name], width=16).grid(row=i, column=0, sticky='w')
            ttk.Entry(self.fields, textvariable=self.vars[name], width=10).grid(row=i, column=1)
            self.limit_labels[name] = ttk.Label(self.fields, width=30)
            self.limit_labels[name].grid(row=i, column=2, sticky='w', padx=6)
        self.check()

    def num(self, name):
        try:
            return float(self.vars[name].get())
        except ValueError:
            return None

    # ------------------------------------------------------------ limits
    def limits(self):
        """(min, max) for every value of the current shape, given the current centre."""
        x0, x1, y0, y1 = self.area
        lim = {'center_x': (x0, x1), 'center_y': (y0, y1), 'sides': (3, 6)}
        cx, cy = self.num('center_x'), self.num('center_y')
        if cx is None or cy is None or not (x0 <= cx <= x1 and y0 <= cy <= y1):
            room = 0.0                                  # centre invalid -> no size allowed
        else:
            room = min(cx - x0, x1 - cx, cy - y0, y1 - cy)   # distance to the nearest edge
        r_max = round(SAFETY * room, 3)
        lim['circle_radius'] = (MIN_R, r_max)
        lim['square_size'] = (2 * MIN_R + 0.02, round(2 * SAFETY * room, 3))
        if self.shape() == 'square_circle':
            s = self.num('square_size') or 0.0
            lim['circle_radius'] = (MIN_R, round(min(r_max, s / 2 - 0.01), 3))
        return lim

    def check(self):
        """Colour every field: OK, or the allowed range in red. Returns True if all OK."""
        if not self.limit_labels:
            return bool(self.shape()) and not SHAPES[self.shape()]
        lim, ok = self.limits(), True
        for name, label in self.limit_labels.items():
            lo, hi = lim[name]
            v = self.num(name)
            fmt = '{:.0f}' if name == 'sides' else '{:.3f}'
            rng = f'{fmt.format(lo)} .. {fmt.format(hi)}'
            if v is None or not lo <= v <= hi or (name == 'sides' and v != int(v)):
                label.configure(text=f'allowed {rng}', foreground='red')
                ok = False
            else:
                label.configure(text=f'max {fmt.format(hi)}', foreground='green')
        self.draw_btn.state(['!disabled'] if ok else ['disabled'])     # the lock
        return ok

    # ------------------------------------------------------------ random
    def randomize(self):
        """Good random values: size 50-80 % of the safe maximum, centre near the middle."""
        x0, x1, y0, y1 = self.area
        names = SHAPES[self.shape()]
        if not names:
            return
        biggest = SAFETY * min(x1 - x0, y1 - y0) / 2           # largest radius anywhere
        r = random.uniform(0.5, 0.8) * biggest
        r = max(r, MIN_R + 0.01)
        need = r / SAFETY                                      # free space needed around centre
        # centre: near the middle of the area (easiest for the Panda and the UR5 camera)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        cx = random.uniform(max(x0 + need, mx - 0.25 * (x1 - x0)), min(x1 - need, mx + 0.25 * (x1 - x0)))
        cy = random.uniform(max(y0 + need, my - 0.15), min(y1 - need, my + 0.15))
        self.vars['center_x'].set(f'{cx:.3f}')
        self.vars['center_y'].set(f'{cy:.3f}')
        self.vars['circle_radius'].set(f'{r:.3f}')
        self.vars['sides'].set(str(random.randint(3, 6)))
        if 'square_size' in names:
            s = 2 * r
            self.vars['square_size'].set(f'{s:.3f}')
            self.vars['circle_radius'].set(f'{s * random.uniform(0.25, 0.4):.3f}')
        self.check()

    # ------------------------------------------------------------ run commands
    def commands(self):
        shape = self.shape()
        draw = ['ros2', 'run', 'pattern_drawer', 'draw_node'] + SIM + ['-p', f'shape:={shape}']
        for name in SHAPES[shape]:
            val = str(int(self.num(name))) if name == 'sides' else f'{self.num(name):.3f}'
            draw += ['-p', f'{name}:={val}']
        cmds = [CLEAR, draw] if self.clear_first.get() else [draw]
        if self.full_run.get():
            cmds += [['ros2', 'run', 'pattern_drawer', 'scan_drawing'] + SIM,
                     ['ros2', 'run', 'pattern_drawer', 'shape_recognizer'],
                     ['ros2', 'run', 'pattern_drawer', 'run_report']]
        return cmds

    def draw(self):
        if self.busy or not self.check():
            return
        self.start(self.commands())

    def clear(self):
        """Only remove the ink (Gazebo strips + RViz trail), no drawing."""
        if not self.busy:
            self.start([CLEAR])

    def start(self, cmds):
        self.busy = True
        self.draw_btn.state(['disabled'])
        self.clear_btn.state(['disabled'])
        self.status.configure(text='Running ...', foreground='orange')
        threading.Thread(target=self.run_all, args=(cmds,), daemon=True).start()

    def run_all(self, cmds):
        """Background thread: one command after the other; stop at the first failure."""
        ok = True
        for cmd in cmds:
            self.log_q.put('\n$ ' + ' '.join(cmd) + '\n')
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, start_new_session=True)
            for line in self.proc.stdout:
                self.log_q.put(line)
            code = self.proc.wait()
            self.proc = None
            if code != 0 and cmd is CLEAR:                 # clearing failed: still draw
                self.log_q.put('*** could not clear the ink - is the new pen_trail running? '
                               '(ros2 service list | grep clear_ink)\n')
                ok = ok and len(cmds) > 1                  # Clear button alone -> Failed
                continue
            if code != 0:
                self.log_q.put(f'*** stopped: exit code {code}\n')
                ok = False
                break
        self.log_q.put(('DONE', ok))

    def stop(self):
        if self.proc is not None:
            os.killpg(self.proc.pid, signal.SIGINT)        # like Ctrl+C in a terminal

    def pump_log(self):
        """Move text from the thread into the log box (Tk must only be touched here)."""
        while not self.log_q.empty():
            item = self.log_q.get()
            if isinstance(item, tuple):
                self.busy = False
                self.clear_btn.state(['!disabled'])
                self.status.configure(text='Finished' if item[1] else 'Failed - see log',
                                      foreground='green' if item[1] else 'red')
                self.check()
                continue
            self.log.configure(state='normal')
            self.log.insert('end', item)
            self.log.see('end')
            self.log.configure(state='disabled')
        self.after(100, self.pump_log)


def main():
    DrawGui().mainloop()


if __name__ == '__main__':
    main()