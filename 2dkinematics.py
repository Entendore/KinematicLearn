#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinematics Lab 2D v4.0 — the complete interactive FK/IK course.

FEATURES
========
* 32 guided lessons with auto-checked challenges (progress saved).
* Quiz tab: auto-generated FK / workspace / max-reach / absolute-angle
  questions + a 13-question concept bank, streak scoring, canvas preview.
* FK mode : sliders / grab joints / mirror / joint limits / undo (Ctrl+Z).
* IK mode : 6 solvers (analytic, CCD, Jᵀ, J⁺, DLS, FABRIK), race mode with
            EE-path stats, stepping, DLS λ tuning, obstacles, joint limits,
            multi-start ×24, 5 trace shapes, 2nd-elbow overlay, pin poses.
* Demos   : joint-space vs task-space interpolation (animated A→B).
* Tools   : solver benchmark table (success / iters / path / µs-per-iter),
            CSV + PNG export, scene save/load (incl. pins), preferences.
* Null-space objectives: jiggle / joint-centering / max-manipulability.
* Canvas  : pan (middle-drag), zoom (wheel), grid coordinates, hover
            reachability readout, tolerance ring, workspace heatmap,
            manipulability ellipse, joint-limit pies, angle arcs.

Requires:  pip install PySide6 numpy
Run:       python kinematics_lab.py
"""

import json
import math
import statistics
import sys
import time
from collections import deque

import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, QSettings
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QImage,
                           QKeySequence, QPainter, QPainterPath, QPalette,
                           QPen, QPixmap, QPolygonF, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                               QDoubleSpinBox, QDialog, QFileDialog, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QMainWindow, QMenu, QMessageBox,
                               QPushButton, QRadioButton, QScrollArea,
                               QSlider, QSpinBox, QSplitter, QTableWidget,
                               QTabWidget, QTableWidgetItem, QTextBrowser,
                               QVBoxLayout, QWidget)

D2R = math.pi / 180.0
R2D = 180.0 / math.pi
APP_VERSION = "4.0"

LINK_COLORS = ["#5b8cff", "#3ecf8e", "#f7b32b", "#ef6461", "#a78bfa", "#2dd4bf"]
DEF_LENS = [110, 90, 75, 65, 55, 50]
DEF_ANGS = [55, -35, 25, -15, 10, 0]

FK_DESC = ("Forward kinematics: set joint angles θ — the pose follows from "
           "geometry. Drag the sliders, or grab any joint / the end effector "
           "directly on the canvas. Ctrl+Z undoes.")
IK_DESC = ("Inverse kinematics: drag the target marker — a solver searches for "
           "joint angles that put the end effector on the target. Use Step to "
           "advance one iteration and watch the error shrink.")


# ----------------------------------------------------------------------------
# Kinematics helpers
# ----------------------------------------------------------------------------
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def fk_positions(angles, lengths):
    """FK: joint positions for relative angles. Returns n+1 (x, y) tuples."""
    pts = [(0.0, 0.0)]
    x = y = 0.0
    acc = 0.0
    for th, L in zip(angles, lengths):
        acc += th
        x += L * math.cos(acc)
        y += L * math.sin(acc)
        pts.append((x, y))
    return pts


def jacobian_of(pts, n):
    """2 x n geometric Jacobian of the end effector."""
    J = np.zeros((2, n))
    ex, ey = pts[-1]
    for i in range(n):
        J[0, i] = -(ey - pts[i][1])
        J[1, i] = ex - pts[i][0]
    return J


def positions_to_angles(pts, n):
    out, prev = [], 0.0
    for i in range(n):
        a = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
        out.append(wrap(a - prev))
        prev = a
    return out


def analytic_angles(lengths, target, elbow_up=True):
    """Closed-form 2-link IK. Returns [a1, a2] or None if degenerate."""
    l1, l2 = lengths[0], lengths[1]
    d = math.hypot(target[0], target[1])
    if d < 1e-9 and abs(l1 - l2) < 1e-9:
        return None
    d = min(max(d, abs(l1 - l2)), l1 + l2)
    c2 = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    a2 = math.acos(max(-1.0, min(1.0, c2)))
    if elbow_up:
        a2 = -a2
    a1 = (math.atan2(target[1], target[0])
          - math.atan2(l2 * math.sin(a2), l1 + l2 * math.cos(a2)))
    return [wrap(a1), wrap(a2)]


def star_poly(cx, cy, R=12.0, r=5.0, points=5):
    poly = QPolygonF()
    for i in range(points * 2):
        rad = R if i % 2 == 0 else r
        a = math.pi / 2 + i * math.pi / points
        poly.append(QPointF(cx - rad * math.cos(a), cy - rad * math.sin(a)))
    return poly


def star_vertices(cx, cy, R, n=5):
    """Pentagram vertex order: successive vertices are 144° apart."""
    return [(cx + R * math.cos(math.pi / 2 + k * 4 * math.pi / n),
             cy + R * math.sin(math.pi / 2 + k * 4 * math.pi / n))
            for k in range(n)]


def path_point(kind, t, R):
    """Parametric trace paths, t in [0, 2pi)."""
    if kind == 0:                                    # circle
        return (0.55 * R * math.cos(t), 0.15 * R + 0.55 * R * math.sin(t))
    if kind == 1:                                    # line (back & forth)
        return (0.70 * R * math.cos(t), 0.25 * R)
    if kind == 2:                                    # lissajous
        return (0.60 * R * math.sin(2 * t), 0.45 * R * math.sin(3 * t))
    if kind == 3:                                    # square
        a = 0.45 * R
        u = (t / (2 * math.pi)) * 8 * a
        side, f = int(u // a) % 4, u % a
        return [(-a + f, -a), (a, -a + f), (a - f, a), (-a, a - f)][side]
    verts = star_vertices(0.0, 0.28 * R, 0.62 * R)   # star (chord path)
    seg = (t / (2 * math.pi)) * 5
    k, f = int(seg) % 5, seg - math.floor(seg)
    x0, y0 = verts[k]
    x1, y1 = verts[(k + 1) % 5]
    return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)


def app_icon():
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#1b1e26"))
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    pts = [(12, 50), (30, 26), (44, 34), (54, 16)]
    for i in range(3):
        pen = QPen(QColor(LINK_COLORS[i]), 7)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(*pts[i]), QPointF(*pts[i + 1]))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#ff5d5d"))
    p.drawEllipse(QPointF(54, 16), 6, 6)
    p.end()
    return QIcon(pm)


# ----------------------------------------------------------------------------
# IK solvers
# ----------------------------------------------------------------------------
class SolverBase:
    max_iter = 4000

    def __init__(self):
        self.angles, self.lengths = [], []
        self.target = np.zeros(2)
        self.iterations = 0
        self.tolerance = 2.0
        self.obstacles = []
        self.avoid = False
        self.use_limits = False
        self.limits = []
        self.ee_path = 0.0
        self.time_us = 0.0
        self._last_ee = None

    def reset(self, angles, lengths, target, tolerance):
        self.angles = [wrap(a) for a in angles]
        self.lengths = list(lengths)
        self.target = np.asarray(target, float)
        self.tolerance = tolerance
        self.iterations = 0
        self.ee_path = 0.0
        self.time_us = 0.0
        self._last_ee = fk_positions(self.angles, self.lengths)[-1]
        self._init_extra()

    @property
    def us_per_it(self):
        return self.time_us / self.iterations if self.iterations else 0.0

    def _init_extra(self):
        pass

    def error(self):
        pts = fk_positions(self.angles, self.lengths)
        return math.hypot(pts[-1][0] - self.target[0],
                          pts[-1][1] - self.target[1])

    # template method: subclasses implement _step, base handles constraints
    def step(self):
        t0 = time.perf_counter()
        self._step()
        self._post()
        self.iterations += 1
        pts = fk_positions(self.angles, self.lengths)
        ee = pts[-1]
        if self._last_ee is not None:
            self.ee_path += math.hypot(ee[0] - self._last_ee[0],
                                       ee[1] - self._last_ee[1])
        self._last_ee = ee
        self.time_us += (time.perf_counter() - t0) * 1e6
        return math.hypot(ee[0] - self.target[0], ee[1] - self.target[1])

    def _step(self):
        raise NotImplementedError

    def _post(self):
        if self.use_limits and self.limits:
            for i, (lo, hi) in enumerate(self.limits):
                if hi - lo < 2 * math.pi - 1e-9:
                    self.angles[i] = max(lo, min(hi, wrap(self.angles[i])))
        if self.avoid and self.obstacles:
            self._avoid()

    def _avoid(self):
        """Push joints out of obstacles, preserving link lengths."""
        n = len(self.lengths)
        pts = [np.array(p) for p in fk_positions(self.angles, self.lengths)]
        for j in range(1, n + 1):
            for (ox, oy, r) in self.obstacles:
                v = pts[j] - np.array([ox, oy])
                d = float(np.linalg.norm(v))
                if d < r + 8.0:
                    if d < 1e-6:
                        v, d = np.array([1.0, 0.0]), 1.0
                    pts[j] = np.array([ox, oy]) + v * ((r + 8.0) / d)
            for i in range(j, n):                     # refit downstream
                v = pts[i + 1] - pts[i]
                nn = float(np.linalg.norm(v))
                if nn > 1e-9:
                    pts[i + 1] = pts[i] + v * (self.lengths[i] / nn)
        self.angles = positions_to_angles(pts, n)


class AnalyticSolver(SolverBase):
    """Closed-form 2-link IK via the law of cosines."""
    elbow_up = True

    def _step(self):
        a = analytic_angles(self.lengths, self.target, self.elbow_up)
        if a is not None:
            self.angles = a


class CCDSolver(SolverBase):
    """CCD with incremental downstream rotation (no per-joint full FK)."""
    def _step(self):
        n = len(self.angles)
        tx, ty = self.target
        pts = [np.array(p) for p in fk_positions(self.angles, self.lengths)]
        for i in range(n - 1, -1, -1):
            jx, jy = pts[i]
            ex, ey = pts[-1]
            d = wrap(math.atan2(ty - jy, tx - jx)
                     - math.atan2(ey - jy, ex - jx))
            if abs(d) < 1e-12:
                continue
            self.angles[i] = wrap(self.angles[i] + d)
            c, s = math.cos(d), math.sin(d)
            for k in range(i + 1, n + 1):          # rotate downstream in place
                dx, dy = pts[k][0] - jx, pts[k][1] - jy
                pts[k] = (jx + dx * c - dy * s, jy + dx * s + dy * c)


class JacobianSolver(SolverBase):
    """Delta-theta from the 2xn Jacobian: transpose / pseudo-inv / DLS."""
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.max_dq = 0.4
        self.lam = 0.8

    def _step(self):
        n = len(self.angles)
        pts = fk_positions(self.angles, self.lengths)
        e = self.target - np.array(pts[-1])
        err = float(np.linalg.norm(e))
        J = jacobian_of(pts, n)

        if self.variant == "transpose":
            g = J.T @ e
            norm = np.linalg.norm(g)
            if norm < 1e-12:
                return
            ex, ey = pts[-1]
            r_avg = max(sum(math.hypot(ex - pts[i][0], ey - pts[i][1])
                            for i in range(n)) / n, 1.0)
            mag = min(self.max_dq, err / r_avg)
            dq = g * (mag / norm)
        elif self.variant == "pinv":
            dq = np.linalg.pinv(J) @ e
        else:
            dq = J.T @ np.linalg.solve(J @ J.T + self.lam ** 2 * np.eye(2), e)

        m = np.abs(dq).max()
        if m > self.max_dq:
            dq *= self.max_dq / m
        self.angles = [wrap(a + dq[i]) for i, a in enumerate(self.angles)]


class FABRIKSolver(SolverBase):
    """Forward-And-Backward Reaching IK: works on positions."""
    def _init_extra(self):
        self.pts = [np.array(p)
                    for p in fk_positions(self.angles, self.lengths)]
        self.root = self.pts[0].copy()

    def _step(self):
        p, d, n, t = self.pts, self.lengths, len(self.lengths), self.target
        if np.linalg.norm(t - self.root) > sum(d):
            for i in range(n):                            # unreachable: stretch
                r = np.linalg.norm(t - p[i])
                if r > 1e-9:
                    p[i + 1] = p[i] + (t - p[i]) * (d[i] / r)
        else:
            p[n] = t.copy()                               # backward pass
            for i in range(n - 1, -1, -1):
                r = np.linalg.norm(p[i] - p[i + 1])
                if r > 1e-9:
                    p[i] = p[i + 1] + (p[i] - p[i + 1]) * (d[i] / r)
            p[0] = self.root.copy()                       # forward pass
            for i in range(n):
                r = np.linalg.norm(p[i + 1] - p[i])
                if r > 1e-9:
                    p[i + 1] = p[i] + (p[i + 1] - p[i]) * (d[i] / r)
        self.angles = positions_to_angles(p, n)


SOLVER_DEFS = [
    ("Analytic (2-link)", lambda: AnalyticSolver(), "#f7b32b"),
    ("CCD",               lambda: CCDSolver(),      "#ef6461"),
    ("Jac Transpose",     lambda: JacobianSolver("transpose"), "#a78bfa"),
    ("Jac Pseudo-inv",    lambda: JacobianSolver("pinv"),      "#2dd4bf"),
    ("Jac DLS",           lambda: JacobianSolver("dls"),       "#f78fb3"),
    ("FABRIK",            lambda: FABRIKSolver(),   "#7bd88f"),
]


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
class ErrorChart(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.hist = deque(maxlen=460)
        self.setMinimumHeight(104)
        f = QFont(self.font())
        f.setPointSize(9)
        self.setFont(f)

    def add(self, e):
        self.hist.append(e)
        self.update()

    def clear(self):
        self.hist.clear()
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#14161c"))
        p.setPen(QPen(QColor("#2b3140"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1,
                                 self.height() - 1), 6, 6)
        if len(self.hist) < 2:
            p.setPen(QPen(QColor("#7b8296")))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "error plot — run the solver to see convergence")
            return
        tol = self.win.tol_spin.value()
        vals = list(self.hist)
        lo = math.log10(0.05)
        hi = math.log10(max(max(vals), tol, 1.0)) + 0.15
        x0, y0, W, H = 10, 20, self.width() - 20, self.height() - 34

        def Y(v):
            lv = math.log10(max(v, 0.05))
            return y0 + H * (1.0 - (lv - lo) / (hi - lo))

        yt = Y(tol)
        p.setPen(QPen(QColor(62, 207, 142, 160), 1, Qt.DashLine))
        p.drawLine(QPointF(x0, yt), QPointF(x0 + W, yt))
        p.setPen(QPen(QColor("#3ecf8e")))
        p.drawText(QPointF(x0 + 2, yt - 4), f"tol {tol:.1f}")
        poly = QPolygonF([QPointF(x0 + i * W / (len(vals) - 1), Y(v))
                          for i, v in enumerate(vals)])
        p.setPen(QPen(QColor("#ffb347"), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(poly)
        p.setPen(QPen(QColor("#c8cddc")))
        p.drawText(QPointF(x0, 14), f"error {vals[-1]:.2f} u (log scale)")
        p.drawText(QPointF(x0 + W - 54, 14), f"n = {len(vals)}")


class AnglePlot(QWidget):
    """Rolling history of every joint angle, colour-coded per joint."""
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.hist = deque(maxlen=520)
        self.setMinimumHeight(92)
        f = QFont(self.font())
        f.setPointSize(8)
        self.setFont(f)

    def add(self, angles):
        self.hist.append([wrap(a) for a in angles])
        self.update()

    def clear(self):
        self.hist.clear()
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#14161c"))
        p.setPen(QPen(QColor("#2b3140"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1,
                                 self.height() - 1), 6, 6)
        p.setPen(QPen(QColor("#c8cddc")))
        p.drawText(QPointF(10, 13), "joint angles θᵢ over time")
        vals = list(self.hist)
        if len(vals) < 2:
            p.setPen(QPen(QColor("#7b8296")))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "move anything to record angle history")
            return
        n = len(vals[-1])
        W, H = self.width() - 20, self.height() - 30
        x0, y0 = 10, 24
        cy = y0 + H / 2
        p.setPen(QPen(QColor("#39415a"), 1))
        p.drawLine(QPointF(x0, cy), QPointF(x0 + W, cy))
        for j in range(n):
            poly = QPolygonF([
                QPointF(x0 + i * W / (len(vals) - 1),
                        cy - vals[i][j] / math.pi * (H / 2 - 4))
                for i in range(len(vals))])
            p.setPen(QPen(QColor(LINK_COLORS[j % len(LINK_COLORS)]), 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(poly)
            p.setBrush(QColor(LINK_COLORS[j % len(LINK_COLORS)]))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x0 + W + 8, cy - vals[-1][j] / math.pi
                                  * (H / 2 - 4)), 3, 3)
            p.setPen(QPen(QColor(LINK_COLORS[j % len(LINK_COLORS)])))
            p.drawText(QPointF(x0 + W + 14,
                               cy - vals[-1][j] / math.pi * (H / 2 - 4) + 4),
                       f"θ{j+1}")


# ----------------------------------------------------------------------------
# Canvas
# ----------------------------------------------------------------------------
class ArmCanvas(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag = None
        self._pan_start = None
        self.hover = None
        self.setMouseTracking(True)
        self.setMinimumSize(480, 420)
        self.small_font = QFont(self.font())
        self.small_font.setPointSize(9)
        self.ws_img = None
        self.ws_np = None
        self.ws_dirty = True

    # --- transforms -------------------------------------------------------
    def scale(self):
        total = max(sum(self.win.lengths), 1e-6)
        return (min(self.width(), self.height()) * 0.5 - 46.0) / total * self.zoom

    def w2s(self, x, y):
        s = self.scale()
        return QPointF(self.width() / 2 + x * s + self.pan_x,
                       self.height() / 2 - y * s + self.pan_y)

    def s2w(self, pos):
        s = self.scale()
        return ((pos.x() - self.width() / 2 - self.pan_x) / s,
                -(pos.y() - self.height() / 2 - self.pan_y) / s)

    def invalidate_ws(self):
        self.ws_dirty = True

    # --- painting -----------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#14161c"))

        if self.win.show_heat:
            self._draw_heatmap(p)
        if self.win.show_grid:
            self._draw_grid(p)
        if self.win.show_ws:
            self._draw_workspace(p)
        if self.win.mode == "ik" and self.win.trace_cb.isChecked():
            self._draw_guide(p)
        if self.win.show_trail and len(self.win.trail) > 1:
            self._draw_trail(p)
        self._draw_pins(p)
        self._draw_obstacles(p)
        self._draw_base(p)

        demo = self.win.demo_play
        if demo:
            self._draw_demo(p)
        else:
            if self.win.mode == "ik" and self.win.compare_cb.isChecked():
                for _name, colr, s in self.win.ghosts:
                    self._draw_ghost(p, s.angles, colr)
            if self.win.mode == "ik" and self.win.both_cb.isChecked():
                self._draw_alt_elbow(p)

        self._draw_arm(p)
        self._draw_manip(p)

        if self.win.mode == "ik" and not demo:
            self._draw_target(p)
            if self.win.compare_cb.isChecked():
                self._draw_legend(p)
        self._draw_goal(p)
        self._draw_hover(p)

        p.setPen(QPen(QColor("#2b3140"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 8, 8)
        p.setFont(self.small_font)
        p.setPen(QPen(QColor("#565f75")))
        p.drawText(QPointF(w - 52, h - 10), f"{self.zoom*100:.0f}%")

    def _text(self, p, pos, s, color, anchor="left", bold=False):
        f = self.small_font
        if bold:
            f = QFont(f)
            f.setBold(True)
        p.setFont(f)
        p.setPen(QPen(color))
        if anchor == "center":
            wd = p.fontMetrics().horizontalAdvance(s)
            p.drawText(QPointF(pos.x() - wd / 2, pos.y()), s)
        else:
            p.drawText(pos, s)

    # --- heatmap ------------------------------------------------------------
    def _ws_image(self):
        if not self.ws_dirty and self.ws_img is not None:
            return self.ws_img
        L = self.win.lengths
        R = sum(L)
        N = 340
        k = (N / 2 - 3) / R
        n = len(L)
        dens = np.zeros((N, N))
        if n == 2:
            t = np.linspace(-math.pi, math.pi, 150, endpoint=False)
            T1, T2 = np.meshgrid(t, t)
            X = L[0] * np.cos(T1) + L[1] * np.cos(T1 + T2)
            Y = L[0] * np.sin(T1) + L[1] * np.sin(T1 + T2)
        else:
            rng = np.random.default_rng(7)
            A = rng.uniform(-math.pi, math.pi, (30000, n))
            C = np.cumsum(A, axis=1)
            X = (np.asarray(L) * np.cos(C)).sum(axis=1)
            Y = (np.asarray(L) * np.sin(C)).sum(axis=1)
        ix = np.clip(((X + R) * k).astype(int), 0, N - 1)
        iy = np.clip(((R - Y) * k).astype(int), 0, N - 1)
        np.add.at(dens, (iy, ix), 1)
        mx = dens.max()
        alpha = (np.log1p(dens) / math.log1p(max(mx, 1e-9)) * 185)
        alpha = alpha.astype(np.uint8)
        alpha[alpha < 25] = 0
        rgba = np.zeros((N, N, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 96, 140, 255
        rgba[..., 3] = alpha
        self.ws_np = rgba
        self.ws_img = QImage(rgba.data, N, N, 4 * N, QImage.Format_RGBA8888)
        self.ws_dirty = False
        return self.ws_img

    def _draw_heatmap(self, p):
        R = sum(self.win.lengths)
        p1 = self.w2s(-R, R)
        p2 = self.w2s(R, -R)
        p.drawImage(QRectF(p1, p2), self._ws_image())

    # --- scene elements -------------------------------------------------------
    def _draw_grid(self, p):
        s = self.scale()
        cx = self.width() / 2.0 + self.pan_x
        cy = self.height() / 2.0 + self.pan_y
        step = 50.0 * s
        while step < 14.0:
            step *= 5.0
        world = step / s
        p.setPen(QPen(QColor("#20242f"), 1))
        x = cx - math.floor(cx / step) * step
        while x < self.width():
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            x += step
        y = cy - math.floor(cy / step) * step
        while y < self.height():
            p.drawLine(QPointF(0, y), QPointF(self.width(), y))
            y += step
        p.setPen(QPen(QColor("#39415a"), 1.4))
        p.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
        p.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))
        # axis coordinate numbers (world units)
        p.setFont(self.small_font)
        p.setPen(QPen(QColor("#4a5266")))
        m = 1
        while True:
            sx = cx + m * step
            if sx > self.width():
                break
            p.drawText(QPointF(sx + 3, cy + 14), f"{m*world:.0f}")
            m += 1
        m = 1
        while True:
            sx = cx - m * step
            if sx < 0:
                break
            p.drawText(QPointF(sx + 3, cy + 14), f"{-m*world:.0f}")
            m += 1
        m = 1
        while True:
            sy = cy - m * step
            if sy < 0:
                break
            p.drawText(QPointF(cx + 5, sy - 4), f"{m*world:.0f}")
            m += 1
        m = 1
        while True:
            sy = cy + m * step
            if sy > self.height():
                break
            p.drawText(QPointF(cx + 5, sy - 4), f"{-m*world:.0f}")
            m += 1
        p.setPen(QPen(QColor("#565f75")))
        p.drawText(QPointF(self.width() - 16, cy - 8), "x")
        p.drawText(QPointF(cx + 8, 16), "y")

    def _draw_workspace(self, p):
        s = self.scale()
        o = self.w2s(0.0, 0.0)
        total = sum(self.win.lengths)
        p.setPen(QPen(QColor(91, 140, 255, 120), 1.4, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(o, total * s, total * s)
        self._text(p, QPointF(o.x(), o.y() - total * s - 8),
                   f"max reach {total:.0f}", QColor(91, 140, 255, 190), "center")
        if len(self.win.lengths) >= 2:
            rmin = 2 * max(self.win.lengths) - total
            if rmin > 1.0:
                p.setPen(QPen(QColor(247, 179, 43, 110), 1.4, Qt.DashLine))
                p.drawEllipse(o, rmin * s, rmin * s)
                self._text(p, QPointF(o.x(), o.y() + rmin * s + 16),
                           f"min reach {rmin:.0f}", QColor(247, 179, 43, 170),
                           "center")

    def _draw_guide(self, p):
        if not self.win.trace_guide:
            return
        poly = QPolygonF([self.w2s(x, y) for x, y in self.win.trace_guide])
        p.setPen(QPen(QColor(150, 160, 190, 120), 1.4, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(poly)

    def _draw_trail(self, p):
        poly = QPolygonF([self.w2s(x, y) for (x, y) in self.win.trail])
        p.setPen(QPen(QColor(255, 93, 93, 110), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(poly)

    def _draw_pins(self, p):
        for angles, lengths in self.win.pins:
            spts = [self.w2s(x, y) for (x, y) in fk_positions(angles, lengths)]
            pen = QPen(QColor(139, 147, 168, 110), 2.2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for i in range(len(spts) - 1):
                p.drawLine(spts[i], spts[i + 1])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(139, 147, 168, 130))
            p.drawEllipse(spts[-1], 4, 4)
        if self.win.pins:
            self._text(p, QPointF(self.width() - 14, 22),
                       f"📌 {len(self.win.pins)}", QColor("#8b93a8"), "right")

    def _draw_obstacles(self, p):
        for (ox, oy, r) in self.win.obstacles:
            c = self.w2s(ox, oy)
            rs = r * self.scale()
            p.setBrush(QColor(239, 100, 97, 46))
            p.setPen(QPen(QColor("#ef6461"), 2))
            p.drawEllipse(c, rs, rs)
            p.setPen(QPen(QColor(239, 100, 97, 120), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, rs * 0.55, rs * 0.55)
            d = rs * 0.7071
            p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
            p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))

    def _draw_base(self, p):
        o = self.w2s(0.0, 0.0)
        p.setBrush(QColor("#2a3040"))
        p.setPen(QPen(QColor("#8a93a8"), 1.6))
        p.drawEllipse(o, 9, 9)
        p.setPen(QPen(QColor("#565f75"), 1.2))
        p.drawLine(QPointF(o.x() - 16, o.y() + 13), QPointF(o.x() + 16, o.y() + 13))
        p.drawLine(QPointF(o.x() - 10, o.y() + 19), QPointF(o.x() + 10, o.y() + 19))

    def _draw_ghost(self, p, angles, colr, dash=False, alpha=165, width=3.2):
        pts = fk_positions(angles, self.win.lengths)
        spts = [self.w2s(x, y) for (x, y) in pts]
        c = QColor(colr)
        c.setAlpha(alpha)
        pen = QPen(c, width)
        pen.setCapStyle(Qt.RoundCap)
        if dash:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for i in range(len(spts) - 1):
            p.drawLine(spts[i], spts[i + 1])
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        for sp in spts[1:-1]:
            p.drawEllipse(sp, 3.5, 3.5)
        p.drawEllipse(spts[-1], 5, 5)

    def _draw_alt_elbow(self, p):
        """The other 2-link IK branch (mirrored elbow), dashed."""
        if len(self.win.lengths) != 2:
            return
        alt = analytic_angles(self.win.lengths, self.win.target,
                              not self.win.elbow_cb.isChecked())
        if alt is None:
            return
        self._draw_ghost(p, alt, QColor("#f7b32b"), dash=True, alpha=150)

    def _draw_demo(self, p):
        w = self.win
        if not (w.demo_eeA is not None and w.demo_eeB is not None):
            return
        a = self.w2s(*w.demo_eeA)
        b = self.w2s(*w.demo_eeB)
        p.setPen(QPen(QColor(200, 205, 220, 130), 1.4, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawLine(a, b)
        if len(w.demo_trail) > 1:
            poly = QPolygonF([self.w2s(x, y) for x, y in w.demo_trail])
            p.setPen(QPen(QColor(192, 132, 252, 150), 1.8))
            p.drawPolyline(poly)
        if w.demo_ghost is not None:
            self._draw_ghost(p, w.demo_ghost, QColor("#c084fc"))
        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2 - 12)
        self._text(p, mid, "straight = task-space", QColor("#c8cddc"), "center")
        self._text(p, QPointF(self.width() - 14, 40),
                   "violet arm = joint-space interp.", QColor("#c084fc"), "right")

    def _draw_arm(self, p):
        n = len(self.win.lengths)
        spts = [self.w2s(x, y) for (x, y) in self.win.pts]
        s = self.scale()
        for i in range(n):
            a, b = spts[i], spts[i + 1]
            lw = max(6.0, min(22.0, self.win.lengths[i] * s * 0.10))
            pen = QPen(QColor(LINK_COLORS[i % len(LINK_COLORS)]), lw)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawLine(a, b)
            pen2 = QPen(QColor(255, 255, 255, 70), max(1.5, lw * 0.22))
            pen2.setCapStyle(Qt.RoundCap)
            p.setPen(pen2)
            p.drawLine(a, b)
        for i in range(n):
            jp = spts[i]
            p.setBrush(QColor("#11141b"))
            p.setPen(QPen(QColor("#dfe3ee"), 1.8))
            p.drawEllipse(jp, 7, 7)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#dfe3ee"))
            p.drawEllipse(jp, 2, 2)
        if self.win.limits_cb.isChecked():
            self._draw_limit_arcs(p, spts, n)
        ee = spts[-1]
        p.setBrush(QColor("#ff5d5d"))
        p.setPen(QPen(QColor("#ffd7d7"), 1.6))
        p.drawEllipse(ee, 8.5, 8.5)
        p.setPen(QPen(QColor("#ff5d5d"), 1.6))
        p.drawLine(QPointF(ee.x() - 14, ee.y()), QPointF(ee.x() - 5, ee.y()))
        p.drawLine(QPointF(ee.x() + 5, ee.y()), QPointF(ee.x() + 14, ee.y()))
        p.drawLine(QPointF(ee.x(), ee.y() - 14), QPointF(ee.x(), ee.y() - 5))
        p.drawLine(QPointF(ee.x(), ee.y() + 5), QPointF(ee.x(), ee.y() + 14))
        if self.win.show_angles:
            self._draw_angle_arcs(p, spts)
        if self.win.show_labels:
            self._draw_link_labels(p, spts)

    def _draw_limit_arcs(self, p, spts, n):
        lim = self.win.limits_spin.value() * D2R
        cum = 0.0
        r = 15.0
        p.setBrush(QColor(239, 100, 97, 28))
        p.setPen(QPen(QColor(239, 100, 97, 110), 1.0))
        for i in range(n):
            base = 0.0 if i == 0 else cum
            rect = QRectF(spts[i].x() - r, spts[i].y() - r, 2 * r, 2 * r)
            p.drawPie(rect, int((base - lim) * R2D * 16),
                      int(2 * lim * R2D * 16))
            cum += self.win.angles[i]

    def _draw_angle_arcs(self, p, spts):
        n = len(self.win.angles)
        cum, a = [], 0.0
        for th in self.win.angles:
            a += th
            cum.append(a)
        r = 26.0
        for i in range(n):
            a1 = 0.0 if i == 0 else cum[i - 1]
            span = cum[i] - a1
            if abs(span) < 0.02:
                continue
            col = QColor(LINK_COLORS[i % len(LINK_COLORS)])
            p.setPen(QPen(col, 1.6))
            p.setBrush(Qt.NoBrush)
            jx, jy = spts[i].x(), spts[i].y()
            rect = QRectF(jx - r, jy - r, 2 * r, 2 * r)
            path = QPainterPath()
            path.arcMoveTo(rect, a1 * R2D)
            path.arcTo(rect, a1 * R2D, span * R2D)
            p.drawPath(path)
            mid = a1 + span / 2.0
            d = r + 14
            self._text(p, QPointF(jx + math.cos(mid) * d,
                                  jy - math.sin(mid) * d),
                       f"θ{i+1} {self.win.angles[i]*R2D:+.0f}°", col)

    def _draw_link_labels(self, p, spts):
        p.setFont(self.small_font)
        for i in range(len(self.win.lengths)):
            a, b = spts[i], spts[i + 1]
            dx, dy = b.x() - a.x(), b.y() - a.y()
            L = math.hypot(dx, dy) or 1.0
            pos = QPointF((a.x() + b.x()) / 2 - dy / L * 17,
                          (a.y() + b.y()) / 2 + dx / L * 17)
            p.setPen(QPen(QColor(LINK_COLORS[i % len(LINK_COLORS)]).lighter(165)))
            p.drawText(pos, f"L{i+1} {self.win.lengths[i]:.0f}")

    def _draw_manip(self, p):
        if not getattr(self.win, "show_manip", False):
            return
        J = jacobian_of(self.win.pts, len(self.win.lengths))
        try:
            U, S, _Vt = np.linalg.svd(J)
        except Exception:
            return
        s1, s2 = float(S[0]), float(S[1])
        if s1 < 1e-6:
            return
        c = self.w2s(*self.win.pts[-1])
        p.save()
        p.translate(c)
        p.rotate(-math.degrees(math.atan2(U[1, 0], U[0, 0])))
        p.setPen(QPen(QColor(45, 212, 191, 210), 1.6, Qt.DashLine))
        p.setBrush(QColor(45, 212, 191, 22))
        p.drawEllipse(QPointF(0, 0), max(s1 * self.scale(), 2.0),
                      max(s2 * self.scale(), 1.5))
        p.restore()
        p.setBrush(Qt.NoBrush)
        self._text(p, QPointF(c.x() + 14, c.y() - 12),
                   f"σ₁σ₂ = {s1*s2:.0f}", QColor("#2dd4bf"))

    def _draw_target(self, p):
        w = self.win
        t = w.target
        tp = self.w2s(t[0], t[1])
        ee = w.pts[-1]
        ep = self.w2s(ee[0], ee[1])
        err = math.hypot(ee[0] - t[0], ee[1] - t[1])
        if err > 0.8:
            p.setPen(QPen(QColor(200, 205, 220, 90), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawLine(ep, tp)
        # tolerance ring
        tr = max(w.tol_spin.value() * self.scale(), 4.0)
        p.setPen(QPen(QColor(62, 207, 142, 110), 1, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(tp, tr, tr)
        col = QColor("#3ecf8e") if w.point_reachable(t[0], t[1]) else QColor("#ff5d5d")
        p.setPen(QPen(col, 2.2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(tp, 11, 11)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(tp.x() + dx * 6, tp.y() + dy * 6),
                       QPointF(tp.x() + dx * 15, tp.y() + dy * 15))
        if not w.point_reachable(t[0], t[1]):
            self._text(p, QPointF(tp.x(), tp.y() + 28),
                       "out of reach / blocked", QColor("#ff5d5d"), "center")
        elif w.show_labels:
            self._text(p, QPointF(tp.x() + 16, tp.y() - 14),
                       f"({t[0]:.0f}, {t[1]:.0f})", QColor("#8fd3a8"))

    def _draw_goal(self, p):
        g = self.win.goal
        if not g:
            return
        if self.win.mode == "ik" and not self.win.demo_play:
            t = self.win.target
            if math.hypot(g[0] - t[0], g[1] - t[1]) < 3.0:
                return
        gp = self.w2s(*g)
        p.setBrush(QColor(255, 215, 94, 60))
        p.setPen(QPen(QColor("#ffd75e"), 2))
        p.drawPolygon(star_poly(gp.x(), gp.y()))
        tol = max(self.win.goal_tol * self.scale(), 5.0)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 215, 94, 120), 1, Qt.DashLine))
        p.drawEllipse(gp, tol, tol)
        if self.win.ch_done:
            self._text(p, QPointF(gp.x(), gp.y() - tol - 8), "✓ done",
                       QColor("#3ecf8e"), "center", bold=True)

    def _draw_hover(self, p):
        if (self.hover is None or self._drag is not None
                or self.win.mode != "ik" or self.win.demo_play):
            return
        x, y = self.hover
        pos = self.w2s(x, y)
        p.setPen(QPen(QColor(200, 205, 220, 90), 1))
        p.drawLine(QPointF(pos.x() - 9, pos.y()), QPointF(pos.x() + 9, pos.y()))
        p.drawLine(QPointF(pos.x(), pos.y() - 9), QPointF(pos.x(), pos.y() + 9))
        ok = self.win.point_reachable(x, y)
        col = QColor("#3ecf8e") if ok else QColor("#ff5d5d")
        self._text(p, QPointF(pos.x() + 13, pos.y() - 8),
                   f"({x:.0f}, {y:.0f})  d={math.hypot(x, y):.0f}  "
                   f"{'reachable' if ok else 'unreachable'}", col)

    def _draw_legend(self, p):
        x, y = 14.0, 20.0
        w = self.win
        tol = w.tol_spin.value()
        best = min((s.error() for _n, _c, s in w.ghosts), default=0.0)
        p.setFont(self.small_font)
        for name, colr, s in w.ghosts:
            p.setPen(Qt.NoPen)
            p.setBrush(colr)
            p.drawEllipse(QPointF(x + 4, y - 3), 4.5, 4.5)
            if abs(s.error() - best) < 1e-9:
                col = QColor("#3ecf8e")
            elif s.iterations > 500 and s.error() > tol:
                col = QColor("#f7b32b")            # stalled
            else:
                col = QColor("#c8cddc")
            p.setPen(QPen(col))
            p.drawText(QPointF(x + 15, y),
                       f"{name}  err {s.error():5.1f} · it {s.iterations:3d}"
                       f" · path {s.ee_path:4.0f}")
            y += 19

    # --- interaction ----------------------------------------------------------
    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.MiddleButton:
            self._drag = "pan"
            self._pan_start = (pos.x() - self.pan_x, pos.y() - self.pan_y)
            return
        if e.button() != Qt.LeftButton:
            return
        if self.win.demo_play:
            return
        if self.win.mode == "ik":
            self.win.set_target(self.s2w(pos))
            self._drag = "t"
        else:
            pts = fk_positions(self.win.angles, self.win.lengths)
            best_i, best_d = -1, 1e9
            for i, (x, y) in enumerate(pts):
                sp = self.w2s(x, y)
                d = math.hypot(sp.x() - pos.x(), sp.y() - pos.y())
                if d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0 and best_d <= 28:
                self.win.push_undo()
                self._drag = best_i

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag == "pan":
            self.pan_x = pos.x() - self._pan_start[0]
            self.pan_y = pos.y() - self._pan_start[1]
            self.update()
            return
        self.hover = self.s2w(pos)
        if self._drag is None:
            self.update()
            return
        if self.win.demo_play:
            return
        if self._drag == "t":
            wx, wy = self.s2w(pos)
            if e.modifiers() & Qt.ShiftModifier:       # snap target to 10 u grid
                wx, wy = round(wx / 10.0) * 10.0, round(wy / 10.0) * 10.0
            self.win.set_target((wx, wy))
        else:
            j = max(self._drag - 1, 0)
            jx, jy = fk_positions(self.win.angles, self.win.lengths)[j]
            wx, wy = self.s2w(pos)
            want = math.atan2(wy - jy, wx - jx)
            if e.modifiers() & Qt.ShiftModifier:           # snap to 15°
                want = round(want / (15 * D2R)) * (15 * D2R)
            self.win.angles[j] = wrap(want - sum(self.win.angles[:j]))
            self.win.clamp_angles()
            self.win.refresh()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def leaveEvent(self, _e):
        self.hover = None
        self.update()

    def wheelEvent(self, e):
        self.zoom = max(0.35, min(6.0, self.zoom * 1.1 ** (e.angleDelta().y() / 120)))
        self.update()

    def mouseDoubleClickEvent(self, _e):
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.update()

    def contextMenuEvent(self, ev):
        m = QMenu(self)
        wx, wy = self.s2w(ev.pos())
        hit = None
        for i, (ox, oy, r) in enumerate(self.win.obstacles):
            if math.hypot(wx - ox, wy - oy) < r:
                hit = i
                break
        if hit is not None:
            m.addAction("Remove this obstacle",
                        lambda: self.win.remove_obstacle(hit))
        m.addAction("Add obstacle here",
                    lambda: self.win.add_obstacle(
                        wx, wy, float(np.random.uniform(22, 44))))
        m.addAction("Clear all obstacles", self.win.clear_obstacles)
        m.exec(ev.globalPos())


# ----------------------------------------------------------------------------
# Lesson check/setup helpers
# ----------------------------------------------------------------------------
def _chk_stretch_fold(win, st):
    total = sum(win.lengths)
    d = math.hypot(win.pts[-1][0], win.pts[-1][1])
    st["a"] = st.get("a", False) or d >= 0.985 * total
    b = (d <= abs(win.lengths[0] - win.lengths[1]) + 6.0
         if len(win.lengths) == 2 else d <= 0.25 * total)
    st["b"] = st.get("b", False) or b
    msg = (f"{'✓' if st['a'] else '•'} stretch the arm to max reach    "
           f"{'✓' if st['b'] else '•'} fold it nearly closed")
    return (st["a"] and st["b"]), msg


def _elbow_check(w, st):
    tol = w.tol_spin.value()
    if w.err_main() < tol:
        if w.angles[1] < 0:
            st["up"] = True
        elif w.angles[1] > 0:
            st["down"] = True
    msg = (f"on target: {w.err_main():.1f} u · "
           f"{'✓' if st.get('up') else '•'} elbow-up found   "
           f"{'✓' if st.get('down') else '•'} elbow-down found")
    return st.get("up", False) and st.get("down", False), msg


def _sing_check(w, st):
    total = sum(w.lengths)
    d = math.hypot(w.pts[-1][0], w.pts[-1][1])
    wv = w.manip()
    st["a"] = st.get("a", False) or (d >= 0.97 * total and wv < 8.0)
    st["b"] = (st.get("b", False)
               or (d <= abs(w.lengths[0] - w.lengths[1]) + 6.0 and wv < 8.0))
    msg = (f"w = {wv:.1f} · {'✓' if st['a'] else '•'} stretched singularity   "
           f"{'✓' if st['b'] else '•'} folded singularity")
    return st["a"] and st["b"], msg


def _race_check(w, st):
    tol = w.tol_spin.value()
    conv = sum(1 for _n, _c, s in w.ghosts if s.error() < tol)
    total = len(w.ghosts)
    done = (w.solver_combo.currentIndex() == 5 and total > 0 and conv == total)
    return done, (f"solvers converged: {conv}/{total} · "
                  f"main error {w.err_main():.2f} u")


def _null_setup(w):
    w.apply_setup(5, [110, 90, 75, 65, 55], [30, -40, 50, -30, 20],
                  mode="ik", solver=3, live=False)
    w.goal = tuple(np.array(w.pts[-1]))
    w.null_mode.setCurrentIndex(0)


def _null_check(w, st):
    st.setdefault("start", list(w.angles))
    st["moved"] = max(st.get("moved", 0.0),
                      max((abs(a - s) for a, s in zip(w.angles, st["start"])),
                          default=0.0))
    if math.dist(w.pts[-1], w.goal) < 4.0:
        st["held"] = st.get("held", 0) + 1
    else:
        st["held"] = 0
    done = st["moved"] >= 35 * D2R and st["held"] >= 50
    msg = (f"run <b>Null-space jiggle</b> · joints moved "
           f"{st['moved']*R2D:.0f}°/35° · EE held {st['held']}/50 frames")
    return done, msg


def _null2_setup(w):
    w.apply_setup(5, [110, 90, 75, 65, 55], [70, -15, 60, 20, -40],
                  mode="ik", solver=3, live=False)
    w.goal = tuple(np.array(w.pts[-1]))
    w.null_mode.setCurrentIndex(2)


def _null2_check(w, st):
    st.setdefault("start_w", max(w.manip(), 1e-6))
    st.setdefault("start", list(w.angles))
    moved = max((abs(a - s) for a, s in zip(w.angles, st["start"])), default=0.0)
    near = math.dist(w.pts[-1], w.goal) < 4.0
    if w.null_on and near:
        st["held"] = st.get("held", 0) + 1
    else:
        st["held"] = 0
    ratio = w.manip() / st["start_w"]
    done = st["held"] >= 40 and moved >= 10 * D2R and ratio >= 1.2
    msg = (f"null-space objective <b>max manip.</b> · w ×{ratio:.2f} (goal ≥1.2)"
           f" · moved {moved*R2D:.0f}°/10° · hold {st['held']}/40")
    return done, msg


def _obstacle_setup(w):
    total = sum(w.lengths)
    tx, ty = 0.92 * total * math.cos(0.5), 0.92 * total * math.sin(0.5)
    w.apply_setup(3, [110, 90, 75], [15, 20, -10], mode="ik", solver=1,
                  target=(tx, ty), goal=(tx, ty),
                  obstacles=[(tx * .45, ty * .45, 34)],
                  avoid=True, live=False)


def _needle_setup(w):
    tx, ty = 205.0, 40.0
    w.apply_setup(3, [110, 90, 75], [10, 15, -5], mode="ik", solver=1,
                  target=(tx, ty), goal=(tx, ty),
                  obstacles=[(95, 60, 30), (95, -60, 30)], avoid=True,
                  live=False)


def _multi_setup(w):
    w.apply_setup(3, [110, 90, 75], [15, 20, -10], mode="ik", solver=1,
                  target=(180, 60), goal=(180, 60),
                  obstacles=[(78, 42, 30), (96, -28, 26)], avoid=True,
                  live=False)
    w.last_multistart = False


def _multi_check(w, st):
    done = (w.last_multistart and len(w.obstacles) >= 2
            and w.err_main() < w.tol_spin.value())
    return done, (f"{'✓ multistart used · ' if w.last_multistart else ''}"
                  f"{len(w.obstacles)} obstacles · err {w.err_main():.2f} u — "
                  f"press <b>Multi-start ×24</b>")


def _limits_check(w, st):
    lim = w.limits_spin.value() * D2R
    ok_lim = (w.limits_cb.isChecked()
              and all(abs(wrap(a)) <= lim + 1e-6 for a in w.angles))
    done = ok_lim and w.err_main() < w.tol_spin.value()
    worst = max(abs(wrap(a)) for a in w.angles) * R2D
    return done, (f"limits ±{w.limits_spin.value()}° · worst joint "
                  f"{worst:.0f}° · err {w.err_main():.2f} u")


def _interp_setup(w):
    w.apply_setup(2, [110, 90], [40, -30], mode="ik", solver=5, live=False)
    w.set_demo_a()
    w.angles = [120 * D2R, 60 * D2R]
    w.refresh()
    w.set_demo_b()
    w.angles = [40 * D2R, -30 * D2R]
    w.refresh()
    w.btn_demo_play.setEnabled(True)
    w.btn_demo_play.setText("▶ Play — click me!")
    w.demo_bounces = 0


def _interp_check(w, st):
    done = w.demo_bounces >= 2
    return done, (f"press <b>▶ Play</b> and watch one full round trip · "
                  f"bounces {w.demo_bounces}/2 — violet arm interpolates in "
                  f"joint space, main arm in task space")


def _trace_check(w, st):
    st.setdefault("last", 0.0)
    st.setdefault("prog", 0.0)
    d = w.trace_t - st["last"]
    if d < -math.pi:
        d += 2 * math.pi
    st["prog"] += max(d, 0.0)
    st["last"] = w.trace_t
    done = w.trace_cb.isChecked() and st["prog"] >= 5.9
    return done, (f"trace progress {min(st['prog'] / (2 * math.pi), 1.0) * 100:.0f}%"
                  + ("" if w.trace_cb.isChecked() else " — trace is off!"))


def _speed_check(w, st):
    tol = w.tol_spin.value()
    it = w.solver.iterations
    done = w.err_main() < tol and 0 < it <= 35
    return done, (f"err {w.err_main():.2f} u in {it} iters (goal ≤ 35) — "
                  f"try FABRIK or Jac Pseudo-inv!")


def _tour_setup(w):
    L = [110, 90, 75, 65]
    w.apply_setup(4, L, [40, -30, 25, -10], mode="ik", solver=3, live=True)
    total = sum(L)
    wps = star_vertices(0.0, 0.10 * total, 0.45 * total)
    w.lesson_state["wps"] = wps
    w.lesson_state["idx"] = 0
    w.goal = wps[0]
    w.set_target(wps[0])


def _tour_check(w, st):
    wps = st.get("wps") or []
    if not wps:
        return False, "—"
    tol = w.tol_spin.value()
    if w.err_main() < tol:
        idx = st.get("idx", 0) + 1
        st["idx"] = idx
        if idx >= len(wps):
            return True, "tour complete — all 5 stars visited ✓"
        w.goal = wps[idx]
        w.set_target(wps[idx])
    idx = st.get("idx", 0)
    return False, (f"star {idx+1}/5 — park the end effector exactly on the "
                   f"gold star (error < {tol:.1f} u, then it auto-advances)")


def _fold_check(w, st):
    """Foldability lesson: reach a point at the very base with 3 links."""
    Lmax = max(w.lengths)
    rest = sum(w.lengths) - Lmax
    foldable = Lmax <= rest + 1e-6
    done = (foldable and len(w.lengths) == 3
            and w.err_main() < w.tol_spin.value())
    return done, (f"foldability: max link {Lmax:.0f} vs rest {rest:.0f} → "
                  f"min reach {'0 ✓' if foldable else f'{Lmax-rest:.0f} ✗'}"
                  f" · err {w.err_main():.2f} u")


def _jaccol_check(w, st):
    """Reading the Jacobian: make column θ1 dominate column θ2 in x."""
    if len(w.lengths) != 3:
        return False, "keep 3 links for this lesson"
    J = jacobian_of(w.pts, 3)
    c1, c2 = abs(J[0, 0]), abs(J[0, 1])
    wm = w.manip()
    ok = c1 > 2.2 * c2 + 5.0 and wm > 25.0
    return ok, (f"x-levers: θ₁→{c1:6.1f}  θ₂→{c2:6.1f} (need θ₁ > 2.2·θ₂+5) "
                f"· w={wm:.0f} — raise joint 1 high, then lay links 2–3 level")


def _tol_cost_check(w, st):
    tol = w.tol_spin.value()
    err = w.err_main()
    if err <= tol:
        if tol >= 5.0:
            st["coarse"] = True
        if tol <= 1.0 + 1e-9:
            st["fine"] = True
    done = st.get("coarse", False) and st.get("fine", False)
    msg = (f"{'✓' if st.get('coarse') else '•'} solve with tolerance ≥ 5    "
           f"{'✓' if st.get('fine') else '•'} solve with tolerance ≤ 1 "
           f"(now: {tol:.1f} · err {err:.2f} · {w.solver.iterations} it)")
    return done, msg


def _mirror_check(w, st):
    if len(w.lengths) != 2:
        return False, "keep 2 links for this lesson"
    ee = w.pts[-1]
    st["a"] = st.get("a", False) or math.dist(ee, (90, 60)) < 8.0
    st["b"] = st.get("b", False) or math.dist(ee, (90, -60)) < 8.0
    done = st["a"] and st["b"]
    msg = (f"{'✓' if st['a'] else '•'} reach (90, 60)    "
           f"{'✓' if st['b'] else '•'} reach (90, −60) — try the ⇄ Mirror "
           f"button or X")
    return done, msg


def _marathon_check(w, st):
    st.setdefault("count", 0)
    st.setdefault("armed", True)
    if w.err_main() <= w.tol_spin.value():
        if st["armed"]:
            st["count"] += 1
            st["armed"] = False
    else:
        st["armed"] = True
    done = st["count"] >= 10
    return done, (f"converged targets: {st['count']}/10 — press <b>R</b> "
                  f"(random target) and let it lock on, repeat")


# ----------------------------------------------------------------------------
# Quiz generators + concept bank
# ----------------------------------------------------------------------------
def quiz_gen_fk(rng):
    L = [int(rng.integers(80, 131)), int(rng.integers(80, 131))]
    a = [int(rng.integers(-9, 10)) * 15, int(rng.integers(-9, 10)) * 15]
    x, y = fk_positions([v * D2R for v in a], L)[-1]
    xi, yi = int(round(x)), int(round(y))
    correct = f"({xi}, {yi})"
    seen = {correct}
    distr = []
    for c in [(xi + 38, yi - 24), (xi - 27, yi + 33), (yi, xi),
              (-xi, yi + 12), (xi + 12, -yi), (xi - 45, yi)]:
        s = f"({c[0]}, {c[1]})"
        if s not in seen:
            seen.add(s)
            distr.append(s)
        if len(distr) == 3:
            break
    while len(distr) < 3:
        s = f"({xi + len(distr) * 17 + 9}, {yi - 11})"
        if s not in seen:
            seen.add(s)
            distr.append(s)
    return {
        "q": (f"FK: l₁={L[0]}, l₂={L[1]}, θ₁={a[0]:+d}°, θ₂={a[1]:+d}° "
              "(relative). Where is the end effector (x, y)?"),
        "options": [correct] + distr,
        "correct": 0,
        "explain": (f"Θ₁={a[0]}°, Θ₂={a[0]+a[1]}° → "
                    f"x = {L[0]}cosΘ₁ + {L[1]}cosΘ₂ = {x:.1f},  y = {y:.1f}."),
        "show": [L, a],
    }


def quiz_gen_reach(rng):
    L = [int(rng.integers(70, 141)), int(rng.integers(70, 141))]
    rmax, rmin = L[0] + L[1], abs(L[0] - L[1])
    r = int(rng.integers(max(30, int(rmin * 0.6)), int(rmax * 1.35)))
    ok = rmin <= r <= rmax
    return {
        "q": (f"A 2-link arm has l₁={L[0]}, l₂={L[1]}. A target sits at "
              f"distance r={r} from the base. Reachable?"),
        "options": ["Reachable", "Not reachable"],
        "correct": 0 if ok else 1,
        "explain": (f"Workspace annulus: {rmin} ≤ r ≤ {rmax}. "
                    f"r={r} is " + ("inside." if ok else "outside.")),
    }


def quiz_gen_maxreach(rng):
    L = [int(rng.integers(60, 151)), int(rng.integers(60, 151))]
    correct = str(L[0] + L[1])
    alts = [str(int(math.hypot(L[0], L[1]))), str(abs(L[0] - L[1])),
            str(2 * (L[0] + L[1]))]
    opts, seen = [correct], {correct}
    for a in alts:
        if a not in seen:
            seen.add(a)
            opts.append(a)
    while len(opts) < 4:
        a = str(L[0] + L[1] + len(opts) * 13)
        if a not in seen:
            seen.add(a)
            opts.append(a)
    return {
        "q": f"Maximum reach of a 2-link arm with l₁={L[0]}, l₂={L[1]}?",
        "options": opts, "correct": 0,
        "explain": (f"Fully stretched: r_max = l₁ + l₂ = {L[0]}+{L[1]} = "
                    f"{L[0]+L[1]}. (√(l₁²+l₂²) would be the reach of two "
                    "perpendicular half-links — a classic trap.)"),
    }


def quiz_gen_absang(rng):
    while True:
        a = [int(rng.integers(-6, 7)) * 15 for _ in range(3)]
        if abs(sum(a)) <= 180:
            break
    s = sum(a)
    correct = f"{s:+d}°"
    opts, seen = [correct], {correct}
    for d in (f"{s + 30:+d}°", f"{a[0] + a[1]:+d}°", f"{-s:+d}°",
              f"{s - 15:+d}°"):
        if d not in seen:
            seen.add(d)
            opts.append(d)
    while len(opts) < 4:
        d = f"{s + len(opts) * 45:+d}°"
        if d not in seen:
            seen.add(d)
            opts.append(d)
    return {
        "q": (f"θ₁={a[0]:+d}°, θ₂={a[1]:+d}°, θ₃={a[2]:+d}° (all relative). "
              "Absolute angle of link 3?"),
        "options": opts, "correct": 0,
        "explain": (f"Absolute = running sum: Θ₃ = θ₁+θ₂+θ₃ = "
                    f"{a[0]:+d} {a[1]:+d} {a[2]:+d} = {s:+d}°."),
    }


QUIZ_BANK = [
    ("Which solver finds the EXACT solution instantly (2-link arms only)?",
     ["CCD", "Analytic (law of cosines)", "FABRIK", "Jac Transpose"], 1,
     "The analytic solver uses the law of cosines — closed form, no iterations."),
    ("Δθ = α·Jᵀe is the update rule of…",
     ["Damped least squares", "Jacobian transpose", "FABRIK", "Analytic IK"], 1,
     "Jᵀe is the 'spring pull' gradient — robust but slow near the goal."),
    ("Increasing λ in Damped Least Squares makes steps…",
     ["Bigger and riskier", "Smaller and safer", "Exactly zero",
      "Faster near singularities"], 1,
     "λ² damps large steps — stability up, speed down."),
    ("FABRIK operates directly on…",
     ["Joint angles", "Link positions", "Link lengths", "Velocities"], 1,
     "It pulls positions and re-fits lengths in backward/forward passes."),
    ("CCD rotates joints in which order each sweep?",
     ["Base → tip", "Tip → base", "Random", "All simultaneously"], 1,
     "Cyclic Coordinate Descent works tip-to-base, then repeats."),
    ("Null-space joint motion Δθ = (I − J⁺J)z changes…",
     ["The end-effector position only", "The pose but not the EE position",
      "Link lengths", "Nothing at all"], 1,
     "Self-motion: the pose changes, the task-space position doesn't."),
    ("Manipulability w = σ₁σ₂ ≈ 0 means…",
     ["Maximum agility", "A singularity — some EE directions are impossible",
      "The target is unreachable", "The Jacobian is invertible"], 1,
     "Singular directions appear when links align (stretched or folded)."),
    ("The inner 'min reach' circle of a 2-link arm has radius…",
     ["l₁ + l₂", "|l₁ − l₂|", "l₁ · l₂", "0 always"], 1,
     "Points closer than |l₁−l₂| to the base cannot be reached."),
    ("How many IK solutions does a 2-link arm usually have inside its workspace?",
     ["Exactly one", "Two (elbow up / elbow down)", "Infinitely many",
      "Exactly four"], 1,
     "Two mirror solutions; exactly one on the boundary; none outside."),
    ("Relative joint angles are measured…",
     ["From the x-axis", "From the previous link's direction", "From the base",
      "From vertical"], 1,
     "Absolute direction of link i = θ₁ + … + θᵢ."),
    ("Interpolating LINEARLY in joint space makes the end effector move…",
     ["In a straight line", "Along a curved path", "In a circle",
      "Not at all"], 1,
     "Only task-space (Cartesian) interpolation gives straight EE paths — "
     "watch the interpolation demo!"),
    ("Multi-start IK (random restarts) mainly helps against…",
     ["Singularities", "Local minima", "Joint limits", "Slow FK"], 1,
     "Different random seeds explore different basins; keep the best result."),
    ("Exactly on the max-reach boundary of a 2-link arm, how many IK solutions are there?",
     ["Two, like everywhere inside", "Exactly one (the arm is straight)",
      "None", "Infinitely many"], 1,
     "cos θ₂ = 1 forces θ₂ = 0 — the elbow-up and elbow-down branches merge."),
    ("A 3-link arm can reach its own base exactly when…",
     ["All links are equal", "The longest link ≤ the sum of the other two",
      "The longest link > the sum of the others", "Total length > 100"], 1,
     "Then the chain folds into a triangle touching the base: min reach = 0."),
    ("Negating every joint angle (θ → −θ) mirrors the end effector across…",
     ["The y-axis", "The x-axis", "The line y = x", "Nothing moves"], 1,
     "cos is even, sin is odd: (x, y) → (x, −y). That's the Mirror button."),
]


# ----------------------------------------------------------------------------
# Lesson curriculum (32 lessons)
# ----------------------------------------------------------------------------
LESSONS = [
    dict(
        title="Welcome & quick start",
        challenge="Switch to IK mode and drag the target (or press Step) — "
                  "any solver iteration counts.",
        theory="""<h2>Welcome to Kinematics Lab</h2>
        <p>This course teaches <b>forward kinematics</b> (angles → pose) and
        <b>inverse kinematics</b> (target → angles) by doing. Each lesson sets
        up the arm and checks your goal automatically — the ◎ line turns green
        when you've got it. There is also an endless <b>Quiz tab</b> and a
        <b>Theory tab</b> with all formulas.</p>
        <h3>Layout</h3>
        <ul><li><b>Left</b>: arm canvas + error plot + angle history plot.
        Middle-drag pans, wheel zooms, numbers on the grid are world units.</li>
        <li><b>Lab tab</b>: sliders, solver controls, demos &amp; tools.</li>
        <li><b>Quiz tab</b>: endless practice questions.</li></ul>
        <h3>Your first goal</h3>
        <p>Click the <b>Inverse</b> radio button and drag the target — the arm
        chases it. Then try <b>Step ▸</b> to advance the solver one iteration
        at a time.</p>""",
        setup=lambda w: w.apply_setup(2, [110, 90], [55, -35], mode="fk"),
        check=lambda w, st: (
            w.mode == "ik" and w.solver.iterations >= 1,
            "switch to IK mode and drag the target (or press Step)")),
    dict(
        title="FK — the pose chain",
        challenge="Use the θ sliders (or grab joints) to bring the end "
                  "effector onto the gold star (dashed ring = tolerance).",
        theory="""<h2>Forward kinematics</h2>
        <p>Each joint angle θ<sub>i</sub> is <b>relative to the previous
        link</b>. The world direction of link <i>i</i> is the running sum
        Θ<sub>i</sub> = θ₁+…+θ<sub>i</sub>, so:</p>
        <p><code>x = Σ lᵢ·cos Θᵢ &nbsp;&nbsp; y = Σ lᵢ·sin Θᵢ</code></p>
        <p>FK is a single evaluation of this chain — that's why it's instant.
        Watch the grey <i>(abs …)</i> readout under each slider: it shows the
        cumulative angle. The grid numbers let you verify positions by eye.</p>""",
        setup=lambda w: w.apply_setup(2, [110, 90], [0, 0], mode="fk",
                                      goal=(95, 125), goal_tol=8),
        check=lambda w, st: (
            math.dist(w.pts[-1], w.goal) < w.goal_tol,
            f"distance to goal: {math.dist(w.pts[-1], w.goal):.1f} u")),
    dict(
        title="Relative vs absolute angles",
        challenge="Make link 2 point at absolute 0° (θ₁+θ₂ ≈ 0) while both "
                  "relative angles stay ≥ 20° in magnitude.",
        theory="""<h2>Relative vs absolute</h2>
        <p>A classic source of confusion! Slider θ₂ is measured <b>from link
        1's direction</b>, not from the x-axis. The absolute direction of link
        2 is θ₁+θ₂.</p>
        <p><b>Goal:</b> make both links collinear (link 2 absolute 0°) — but
        with both sliders at least 20° away from zero. Only possible because
        the relative angles cancel.</p>""",
        setup=lambda w: w.apply_setup(2, [110, 90], [60, 40], mode="fk"),
        check=lambda w, st: (
            (abs(wrap(sum(w.angles))) < 3 * D2R
             and abs(w.angles[0]) > 20 * D2R and abs(w.angles[1]) > 20 * D2R),
            f"link-2 absolute angle: {sum(w.angles)*R2D:+.1f}° "
            f"(θ₁={w.angles[0]*R2D:+.0f}°, θ₂={w.angles[1]*R2D:+.0f}°)")),
    dict(
        title="Workspace — stretch & fold",
        challenge="① Stretch the arm to ≥ 98.5% of max reach, then ② fold it "
                  "down to the inner 'min reach' circle.",
        theory="""<h2>The workspace</h2>
        <p>The end effector can only live between two circles:</p>
        <ul><li><b>max reach</b> = l₁+l₂ (arm straight)</li>
        <li><b>min reach</b> = |l₁−l₂| for 2 links (arm folded)</li></ul>
        <p>Targets between the circles are reachable; the rest are not — no IK
        algorithm can fix that. The telemetry shows what fraction of the reach
        disc is actually usable, and the density map shows where random poses
        tend to dump the end effector.</p>""",
        setup=lambda w: w.apply_setup(2, [130, 70], [20, 20], mode="fk"),
        check=_chk_stretch_fold),
    dict(
        title="Analytic IK & elbow states",
        challenge="Reach the star with IK, then find the <b>second</b> "
                  "solution (other elbow) for the same point.",
        theory="""<h2>Closed-form 2-link IK</h2>
        <p>Law of cosines gives θ₂ directly:</p>
        <p><code>cos θ₂ = (x²+y²−l₁²−l₂²) / (2 l₁ l₂)</code></p>
        <p>then θ₁ = atan2(y,x) − atan2(l₂ sin θ₂, l₁+l₂ cos θ₂).</p>
        <p>±θ₂ gives the two mirror solutions: <b>elbow-up</b> and
        <b>elbow-down</b>. Tick <b>show 2nd elbow</b> to see both branches
        overlaid, toggle the checkbox or press <b>X</b> to flip, and use
        <b>📌 Pin pose</b> to freeze a solution for comparison.</p>""",
        setup=lambda w: w.apply_setup(2, [110, 90], [55, -35], mode="ik",
                                      solver=0, target=(70, 150),
                                      goal=(70, 150)),
        check=lambda w, st: _elbow_check(w, st)),
    dict(
        title="The Jacobian & singularities",
        challenge="Trigger both singular configurations: ① arm stretched "
                  "nearly straight, ② arm folded — watch w = σ₁σ₂ crash to ~0.",
        theory="""<h2>The Jacobian</h2>
        <p>Column <i>i</i> of J = ( −(y<sub>e</sub>−y<sub>i</sub>),
        x<sub>e</sub>−x<sub>i</sub> ) — the velocity the end effector gets
        from joint <i>i</i> turning at 1 rad/s. The telemetry shows J live.</p>
        <p><b>Manipulability</b> w = √det(JJᵀ) = σ₁σ₂ measures how many
        directions the arm can still move in. When links align, columns become
        parallel and w → 0: a <b>singularity</b>. The teal ellipse at the end
        effector visualises σ₁ and σ₂.</p>""",
        setup=lambda w: w.apply_setup(2, [110, 90], [15, 25], mode="fk"),
        check=lambda w, st: _sing_check(w, st)),
    dict(
        title="Jacobian pseudo-inverse IK",
        challenge="Use the Jac Pseudo-inv solver (stepping helps) to bring "
                  "the error under tolerance.",
        theory="""<h2>Δθ = J⁺ e</h2>
        <p>The pseudo-inverse J⁺ = Jᵀ(JJᵀ)⁻¹ gives the smallest joint change
        that moves the end effector toward the error vector e. It behaves like
        Newton's method: very fast far from singularities, but it can explode
        near them (later lessons fix that). Watch the error plot dive.</p>
        <p>Try <b>Step ▸</b> repeatedly — each step is one linearised Newton
        iteration. Telemetry also shows µs per iteration: the computational
        price of each solver.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -30, 20],
                                      mode="ik", solver=3, target=(90, 110),
                                      goal=(90, 110)),
        check=lambda w, st: (
            w.solver_combo.currentIndex() == 3 and w.err_main() < w.tol_spin.value(),
            f"pseudo-inv solver · error {w.err_main():.2f} u")),
    dict(
        title="Damped least squares — tune λ",
        challenge="Reach the near-edge target with Jac DLS in ≤ 350 "
                  "iterations. Lower λ = faster but wilder.",
        theory="""<h2>DLS: Δθ = Jᵀ(JJᵀ + λ²I)⁻¹ e</h2>
        <p>Near the workspace edge the arm approaches a singularity and J⁺
        steps blow up. DLS adds λ² to the denominator: big steps get damped,
        small steps barely change — smooth and safe, slightly slower.</p>
        <p><b>Experiment:</b> λ = 0.1 snaps; λ = 2.5 crawls. Find the λ that
        finishes within 350 iterations.</p>""",
        setup=lambda w: w.apply_setup(
            2, [110, 90], [0, 10], mode="ik", solver=4, lam=0.8,
            target=(193 * math.cos(0.45), 193 * math.sin(0.45)),
            goal=(193 * math.cos(0.45), 193 * math.sin(0.45))),
        check=lambda w, st: (
            w.solver_combo.currentIndex() == 4 and w.err_main() < w.tol_spin.value()
            and w.solver.iterations <= 350,
            f"DLS · error {w.err_main():.2f} u in {w.solver.iterations} iters "
            f"(limit 350) · λ={w.lam_spin.value():.1f}")),
    dict(
        title="CCD — cyclic coordinate descent",
        challenge="Reach the target using CCD. Press Step ▸ and watch each "
                  "sweep: joints rotate one-by-one, tip to base.",
        theory="""<h2>CCD</h2>
        <p>The simplest IK there is: for each joint from tip to base, rotate
        it so the end effector aims exactly at the target, repeat until
        close.</p>
        <ul><li>+ no linear algebra, handles any number of links</li>
        <li>− converges slower, path can be jerky</li>
        <li>+ trivially combined with constraints (limits, obstacles)</li></ul>""",
        setup=lambda w: w.apply_setup(4, [110, 90, 75, 65], [30, -25, 20, -10],
                                      mode="ik", solver=1, target=(120, 120),
                                      goal=(120, 120)),
        check=lambda w, st: (
            w.solver_combo.currentIndex() == 1 and w.err_main() < w.tol_spin.value(),
            f"CCD · error {w.err_main():.2f} u after {w.solver.iterations} iters")),
    dict(
        title="FABRIK & the solver race",
        challenge="FABRIK is selected and all six solvers race as ghost arms. "
                  "Get <b>every</b> solver under tolerance.",
        theory="""<h2>FABRIK</h2>
        <p>Instead of angles, FABRIK thinks in <b>positions</b>: pull the end
        effector to the target, re-fit link lengths backward to the root,
        then forward again. Fast, smooth, easy to constrain.</p>
        <h2>The race</h2>
        <p>The legend shows each solver's live error, iteration count and
        <b>EE path length</b>. Orange = stalled (many iterations, still far).
        For the full statistical picture run <b>Tools → Benchmark solvers</b>.
        </p>""",
        setup=lambda w: w.apply_setup(4, [110, 90, 75, 65], [20, -20, 15, -10],
                                      mode="ik", solver=5, target=(110, 130),
                                      goal=(110, 130), compare=True),
        check=lambda w, st: _race_check(w, st)),
    dict(
        title="Redundancy — null-space jiggle",
        challenge="With 5 links, run <b>Null-space jiggle</b>: joints must "
                  "sweep ≥ 35° while the end effector stays within 4 u of "
                  "the star.",
        theory="""<h2>Redundant arms & the null space</h2>
        <p>A 5-link planar arm has 5 joints but only 2 end-effector
        coordinates: 3 dimensions of <b>redundancy</b>. The projector</p>
        <p><code>N = I − J⁺J&nbsp;&nbsp;⇒&nbsp;&nbsp;Δθ = N·z</code></p>
        <p>produces joint motion that does <b>not</b> move the end effector
        (to first order). The jiggle adds a small corrective pull J⁺e to
        cancel drift.</p>""",
        setup=_null_setup,
        check=lambda w, st: _null_check(w, st)),
    dict(
        title="Null space II — posture goals",
        challenge="Same idea, useful objective: switch the null-space mode "
                  "to <b>max manip.</b> and let the arm improve its posture "
                  "(w ×1.2) without moving the end effector.",
        theory="""<h2>Optimising posture in the null space</h2>
        <p>Redundancy is a resource: any objective can be optimised through
        N·z without disturbing the task. A classic one is Yoshikawa's
        <b>manipulability</b> w = √det(JJᵀ):</p>
        <p><code>Δθ = N · ∇w</code> &nbsp;(gradient ascent, projected)</p>
        <p>The arm creeps away from singular poses while the end effector
        stays pinned. Other popular objectives: joint-centering
        (Δθ = −N·θ), distance-to-limits, obstacle clearance.</p>""",
        setup=_null2_setup,
        check=lambda w, st: _null2_check(w, st)),
    dict(
        title="Obstacle avoidance",
        challenge="An obstacle blocks the way. With avoidance ON, get CCD to "
                  "reach the target behind it.",
        theory="""<h2>IK with obstacles</h2>
        <p>Plain IK knows nothing about the world. A cheap, effective fix:
        after every solver iteration, any joint inside an obstacle is
        projected onto its boundary and the rest of the chain is re-fitted.
        Iterative solvers (CCD, FABRIK, Jacobians) absorb this and snake
        around.</p>
        <p>Right-click the canvas to add/remove obstacles. Targets inside an
        obstacle are marked unreachable (red).</p>""",
        setup=lambda w: _obstacle_setup(w),
        check=lambda w, st: (
            w.avoid_cb.isChecked() and len(w.obstacles) >= 1
            and w.err_main() < w.tol_spin.value(),
            f"avoidance on · {len(w.obstacles)} obstacle(s) · "
            f"error {w.err_main():.2f} u")),
    dict(
        title="Thread the needle",
        challenge="Two obstacles with a narrow gap. Steer the arm <b>through "
                  "the gap</b> to the target — no cheating (obstacles must "
                  "stay).",
        theory="""<h2>Local minima & planning</h2>
        <p>Reactive avoidance is greedy: the arm can get stuck against an
        obstacle in a <b>local minimum</b> — every small move makes things
        worse. Tricks that help:</p>
        <ul><li>retract the arm and approach again (new start pose)</li>
        <li>switch solver — CCD and FABRIK get stuck differently</li>
        <li>approach via a via-point (move the target in two stages)</li></ul>
        <p>Real systems combine IK with a <b>path planner</b> (RRT, PRM) that
        handles the global picture.</p>""",
        setup=lambda w: _needle_setup(w),
        check=lambda w, st: (
            w.avoid_cb.isChecked() and len(w.obstacles) >= 2
            and w.err_main() < w.tol_spin.value(),
            f"{len(w.obstacles)} obstacles · error {w.err_main():.2f} u — "
            f"aim through the gap")),
    dict(
        title="Multi-start vs local minima",
        challenge="A pocket of obstacles hides the target. Solve it with the "
                  "<b>Multi-start ×24</b> button — random restarts escape "
                  "the local minimum the single CCD run falls into.",
        theory="""<h2>Random restarts</h2>
        <p>Iterative IK from one seed can only find the basin it starts in.
        <b>Multi-start</b> runs the solver from many random seeds and keeps
        the best result — a brute-force but honest fix for local minima,
        widely used in production when a few hundred milliseconds are
        affordable.</p>
        <p><b>Experiment:</b> first press <b>Step ▸</b> a few times and watch
        CCD grind into the pocket. Then hit Multi-start and compare.</p>""",
        setup=_multi_setup,
        check=lambda w, st: _multi_check(w, st)),
    dict(
        title="Joint limits + IK",
        challenge="With joint limits ±110° enabled, reach the target using "
                  "Jac Transpose. All joints must stay inside their limits.",
        theory="""<h2>Constrained IK</h2>
        <p>Real joints can't spin forever. Here limits are enforced by simple
        <b>clamping</b> after each solver step — crude but effective, and the
        solvers recover on the next iterations. Fancier methods push joints
        away from limits through the null space, or use bounded quadratic
        programming.</p>
        <p>The red pies on the canvas visualise each joint's allowed absolute
        span. Watch the solver lean against a limit and route the remaining
        freedom elsewhere — that's redundancy at work.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -20, 30],
                                      mode="ik", solver=2, target=(185, 95),
                                      goal=(185, 95), limits=110),
        check=lambda w, st: _limits_check(w, st)),
    dict(
        title="Joint-space vs task-space interpolation",
        challenge="Press <b>▶ Play</b> in <i>Demos &amp; tools</i> and watch "
                  "a full round trip between poses A and B.",
        theory="""<h2>Two ways to get from A to B</h2>
        <p><b>Joint-space interpolation</b> (violet arm) lerps the angles.
        Cheap and always smooth — but the end effector wanders along a
        <b>curve</b>.</p>
        <p><b>Task-space (Cartesian) interpolation</b> lerps the EE position
        along the straight grey line and solves IK every frame. Straight-line
        motion — but near singularities joint speeds explode.</p>
        <p>Which to use? Welding/sealing needs straight lines (task-space);
        free-air moves prefer joint-space (faster, safer). Watch the violet
        EE trail bend — that's the whole lesson in one image.</p>""",
        setup=_interp_setup,
        check=lambda w, st: _interp_check(w, st)),
    dict(
        title="Trajectory tracing — circle",
        challenge="Keep trace mode running until the target has travelled a "
                  "full lap of the circle.",
        theory="""<h2>Tracking a moving target</h2>
        <p>Real robots rarely solve IK once — they track a continuous path,
        re-solving every control tick. Because each solve starts from the
        previous solution, small target steps need tiny corrections: this is
        <b>differential IK</b> in disguise.</p>
        <p><b>Try:</b> enable <i>Compare all solvers</i> during the trace and
        see which one tracks smoothly and which zig-zags. Watch the
        angle-history plot ripple.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -30, 20],
                                      mode="ik", solver=3, live=False,
                                      trace=True, trace_kind=0),
        check=lambda w, st: _trace_check(w, st)),
    dict(
        title="Straight-line tracking",
        challenge="Full lap of the back-and-forth line. Notice the arm never "
                  "leaves the line once locked on.",
        theory="""<h2>Why start from the last solution?</h2>
        <p>Re-solving IK from a fixed home pose every tick would make the arm
        jump between the (many) IK solutions. Seeding each solve with the
        previous one keeps the arm on a <b>continuous branch</b> of the
        solution manifold — smooth, predictable motion. This is exactly how
        industrial controllers follow programmed Cartesian paths.</p>
        <p><b>Experiment:</b> pause the trace, drag the target far away, then
        resume — see how it converges back onto the line.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -30, 20],
                                      mode="ik", solver=3, live=False,
                                      trace=True, trace_kind=1),
        check=lambda w, st: _trace_check(w, st)),
    dict(
        title="Draw a square",
        challenge="One complete lap of the square. Corners are where cheap "
                  "solvers overshoot — watch the error plot spike.",
        theory="""<h2>Corners & dynamics</h2>
        <p>At each corner the target velocity flips 90° instantly. A tracking
        solver with any lag overshoots — the error plot spikes, the angle
        plot kinks. Smooth robot paths are usually blended (fillet the
        corners) or time-parameterised with trapezoidal velocity profiles so
        accelerations stay finite.</p>
        <p><b>Try:</b> crank the speed slider up and compare FABRIK vs
        pseudo-inv tracking quality at the corners.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -30, 20],
                                      mode="ik", solver=5, live=False,
                                      trace=True, trace_kind=3),
        check=lambda w, st: _trace_check(w, st)),
    dict(
        title="Speed solve — beat 35 iterations",
        challenge="Reach the target (4 links) in ≤ 35 solver iterations. "
                  "CCD is preselected — you'll want to switch solvers!",
        theory="""<h2>Choosing a solver</h2>
        <p>There is no universally best IK method — it's a trade-off:</p>
        <ul><li><b>Analytic</b>: instant, but only for special arms.</li>
        <li><b>Jacobian pseudo-inv</b>: quadratic-ish convergence, fragile
        near singularities.</li>
        <li><b>DLS</b>: robust, a bit slower.</li>
        <li><b>Transpose</b>: very robust, slowest.</li>
        <li><b>FABRIK</b>: very fast per iteration and smooth — often the
        winner for position-only tasks.</li>
        <li><b>CCD</b>: simple, constraint-friendly, moderate.</li></ul>
        <p>Run <b>Tools → Benchmark solvers</b> to see success rate, median
        iterations, path length and µs/iter measured on this very arm.</p>""",
        setup=lambda w: w.apply_setup(4, [110, 90, 75, 65], [30, -25, 20, -10],
                                      mode="ik", solver=1, target=(140, 140),
                                      goal=(140, 140)),
        check=lambda w, st: _speed_check(w, st)),
    dict(
        title="One link — the protractor",
        challenge="Park the tip of this single 120 u link on the gold star "
                  "at 60°.",
        theory="""<h2>The simplest arm of all</h2>
        <p>With one link there is only FK:</p>
        <p><code>x = l·cos θ&nbsp;&nbsp;&nbsp;y = l·sin θ</code></p>
        <p>The tip lives on a circle of radius l. One joint ↔ one circle —
        this is the atom every bigger arm is built from. Notice there is
        <b>no</b> relative/absolute distinction yet: with a single joint,
        θ is measured straight from the x-axis.</p>
        <p><b>Goal:</b> the star sits at angle 60°, radius 120. Predict the
        coordinates by hand first, then drag the slider.</p>""",
        setup=lambda w: w.apply_setup(1, [120], [0], mode="fk",
                                      goal=(60.0, 103.9), goal_tol=7),
        check=lambda w, st: (
            math.dist(w.pts[-1], w.goal) < w.goal_tol,
            f"tip at r={math.hypot(*w.pts[-1]):.0f}, "
            f"θ={math.degrees(math.atan2(w.pts[-1][1], w.pts[-1][0])):+.1f}°")),
    dict(
        title="Workspace design — fix the arm",
        challenge="The star at (30, 0) is unreachable with l₁=130, l₂=60. "
                  "Fix it using the <b>length sliders</b>, then reach the "
                  "star (2 links only).",
        theory="""<h2>Designing the workspace</h2>
        <p>The inner hole has radius <code>|l₁ − l₂|</code>. Right now that
        is 70 — the star at r=30 sits deep inside it, and no IK on Earth can
        help. But <b>you</b> are the arm designer: shrink the length gap
        until the hole no longer covers the star (|l₁−l₂| ≤ 36 will do), or
        grow L₂ past L₁.</p>
        <p>This is exactly the trade real engineers face: long reach wants
        unequal links; compact work near the base wants nearly equal
        ones.</p>""",
        setup=lambda w: w.apply_setup(2, [130, 60], [10, 20], mode="fk",
                                      goal=(30, 0), goal_tol=6),
        check=lambda w, st: (
            len(w.lengths) == 2
            and math.dist(w.pts[-1], w.goal) < w.goal_tol,
            f"min reach |l₁−l₂| = {abs(w.lengths[0]-w.lengths[1]):.0f}"
            f" (need ≤ 36) · star dist "
            f"{math.dist(w.pts[-1], w.goal):.1f} u")),
    dict(
        title="The inner hole — watching IK fail",
        challenge="The target sits inside the hole (min reach 60). Press "
                  "<b>Space</b> (Solve) and let an iterative solver grind: "
                  "300+ iterations without success completes the lesson.",
        theory="""<h2>Unreachable targets</h2>
        <p>Iterative solvers don't know the workspace — they just chase the
        error downhill. For a target inside the hole they do the only sane
        thing: crawl onto the boundary (|l₁−l₂|) and stall there forever,
        oscillating.</p>
        <p><b>Lesson:</b> IK can never fix geometry. Real systems test
        reachability <i>first</i> (one distance check!) and reject
        impossible targets before wasting cycles. Watch the stall counter in
        the telemetry — and notice the hover readout already told you the
        point was unreachable before you even clicked.</p>""",
        setup=lambda w: w.apply_setup(2, [120, 60], [30, -20], mode="ik",
                                      solver=1, target=(20, 0), goal=(20, 0),
                                      live=False),
        check=lambda w, st: (
            w.solver.iterations > 300 and w.err_main() > 10.0,
            f"iters {w.solver.iterations} · err {w.err_main():.1f} u "
            f"(min reach 60) — press Solve and watch it stall")),
    dict(
        title="Foldable arm — reach the base",
        challenge="Three links: 130, 60, 55. Reach the star at (12, 8) — "
                  "just 14 u from the base!",
        theory="""<h2>The foldability condition</h2>
        <p>For 3+ links the inner hole vanishes when the longest link is no
        longer than all the others combined:</p>
        <p><code>min reach = max(0, L<sub>max</sub> − ΣL<sub>others</sub>)</code></p>
        <p>Here 130 ≤ 60+55 = 115? No — 130 &gt; 115, so there is a small
        hole of radius 15 and the star at r=14 is <i>just</i> inside it.
        Shrink L₁ by a few units and the arm can fold like a collapsible
        ruler, touching its own base. Try it — then push the star even
        closer to (0, 0) mentally and see what the condition demands.</p>""",
        setup=lambda w: w.apply_setup(3, [130, 60, 55], [60, -80, 40],
                                      mode="ik", solver=5, target=(12, 8),
                                      goal=(12, 8)),
        check=_fold_check),
    dict(
        title="Boundary cusp — one solution only",
        challenge="Reach the star parked 0.1 u inside the max-reach circle "
                  "(200). Then toggle Elbow up/down — nothing visibly "
                  "changes. Finish with |θ₂| < 4°.",
        theory="""<h2>Where two solutions become one</h2>
        <p>Inside the workspace a 2-link arm has two mirror solutions.
        Exactly on the boundary circle they <b>merge</b>: the law of cosines
        gives cos θ₂ = 1, so θ₂ = 0 — the arm is dead straight. This single
        point is also a singularity (lesson 6!), and it's why boundary
        tracking feels stiff: the branch choice disappears exactly where the
        Jacobian does.</p>
        <p><b>Goal:</b> the star sits at radius 199.9 (max reach 200). Solve
        analytically and inspect θ₂ in the telemetry — nearly straight, and
        the two mirror branches are visually indistinguishable.</p>""",
        setup=lambda w: w.apply_setup(
            2, [110, 90], [40, -30], mode="ik", solver=0,
            target=(199.9 * math.cos(math.radians(20)),
                    199.9 * math.sin(math.radians(20))),
            goal=(199.9 * math.cos(math.radians(20)),
                  199.9 * math.sin(math.radians(20)))),
        check=lambda w, st: (
            w.err_main() < w.tol_spin.value() and abs(w.angles[1]) < 4 * D2R,
            f"err {w.err_main():.2f} u · θ₂ = {w.angles[1]*R2D:+.1f}° "
            f"(need |θ₂| < 4° — arm nearly straight)")),
    dict(
        title="Reading the Jacobian columns",
        challenge="Arrange the 3-link arm so the θ₁ column's x-lever is over "
                  "2.2× the θ₂ column's — while keeping manipulability "
                  "w > 25. Hint: raise joint 1 high, then lay links 2–3 "
                  "nearly level with it.",
        theory="""<h2>J as a lever list</h2>
        <p>Each column of J answers: <i>if this joint turns, where does the
        EE go?</i> Column θ<sub>i</sub> = ( −(y<sub>e</sub>−y<sub>i</sub>),
        (x<sub>e</sub>−x<sub>i</sub>) ) — it only depends on the vector from
        that joint to the EE.</p>
        <p>So the <b>x-lever</b> of the base joint is the EE's height above
        the <i>base</i>; the x-lever of joint 2 is the EE's height above
        <i>joint 1</i>. To make θ₁ dominate: put the EE far above the base
        but level with joint 1 (links 2–3 horizontal). Watch the live values
        below and the manipulability ellipse — stay clear of the stretched
        singularity.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [15, 20, 10],
                                      mode="fk"),
        check=_jaccol_check),
    dict(
        title="Tolerance vs iteration cost",
        challenge="Solve the same target twice: once with tolerance ≥ 5, "
                  "once with tolerance ≤ 1. Compare the iteration counts in "
                  "the telemetry.",
        theory="""<h2>Precision is not free</h2>
        <p>Convergence is asymptotic: the last decimal place costs as much
        as everything before it. Halving the tolerance can double the
        iterations — and near singularities it's far worse.</p>
        <p><b>Experiment:</b> set tolerance 8, press Space, note the
        iterations. Now tolerance 1, Space again. Real controllers pick the
        loosest tolerance the task allows: ±1 mm is meaningless on a press
        line, life-or-death in electronics assembly.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -30, 20],
                                      mode="ik", solver=4, target=(150, 80),
                                      goal=(150, 80), live=True),
        check=_tol_cost_check),
    dict(
        title="Mirror symmetry",
        challenge="Reach (90, 60), then reach its mirror (90, −60). The ⇄ "
                  "Mirror button (or X) negates every angle at once.",
        theory="""<h2>Reflection is a free solution</h2>
        <p>Negating every joint angle mirrors the whole pose across the
        x-axis: <code>FK(−θ) = (x, −y)</code>. So every solution above the
        axis comes with a twin below it — for free.</p>
        <p>This doubles your IK solution count for symmetric tasks, and it's
        why a mirrored <i>start pose</i> gives a mirrored <i>path</i> in
        trace mode. Try it: mirror now, then run any trace — the arm sweeps
        the reflected region.</p>""",
        setup=lambda w: w.apply_setup(2, [110, 90], [20, 30], mode="fk"),
        check=_mirror_check),
    dict(
        title="Random-target marathon",
        challenge="Converge on <b>10 different random targets</b>. Press R, "
                  "wait for 'converged ✓', repeat.",
        theory="""<h2>IK as a daily tool</h2>
        <p>Everything you've learned compresses into this loop: sample a
        target → check reachability → solve → verify tolerance. Watch the
        patterns across 10 solves:</p>
        <ul><li>targets near the edge take longer (boundary singularity)</li>
        <li>each solve warm-starts from the previous pose — motion stays
        smooth even between distant targets</li>
        <li>the elbow branch persists until a solve forces a flip</li></ul>
        <p>This is the skeleton of every pick-and-place program.</p>""",
        setup=lambda w: w.apply_setup(3, [110, 90, 75], [40, -30, 20],
                                      mode="ik", solver=5, live=True),
        check=_marathon_check),
    dict(
        title="Quiz streak — 5 in a row",
        challenge="Head to the <b>Quiz tab</b> and answer 5 questions "
                  "correctly in a row (streak shown under the score).",
        theory="""<h2>Prove it without the canvas</h2>
        <p>Doing is half the skill; the other half is <i>predicting</i>
        before you touch a slider. The quiz generates endless FK,
        reachability, max-reach, absolute-angle and concept questions — the
        numeric ones can be projected onto the canvas with
        <b>Show pose on canvas</b> to check yourself visually.</p>
        <p><b>Warm-up:</b> l₁=120, l₂=80, θ₁=+45°, θ₂=−45°. Absolute angle
        of link 2? End effector position? Compute first — then verify.</p>""",
        setup=None,
        check=lambda w, st: (
            w.quiz_streak >= 5,
            f"quiz streak {w.quiz_streak}/5 — press <b>Q</b> to jump to the "
            f"quiz tab")),
    dict(
        title="Capstone — the star tour",
        challenge="Visit all 5 stars in order: park on a star, hold until it "
                  "locks in, and the tour auto-advances to the next.",
        theory="""<h2>Waypoint tasks</h2>
        <p>This is what real pick-and-place looks like: a sequence of
        Cartesian goals, each solved to tolerance before advancing. Note how
        warm-starting from the previous waypoint keeps motion smooth — and
        how the arm sometimes takes very different elbow configurations
        between nearby stars.</p>
        <p><b>Congratulations if you got here!</b> You've covered FK, the
        Jacobian, singularities, five iterative IK families, redundancy,
        null-space objectives, constraints, interpolation, workspace design
        and trajectory tracking. The Quiz tab is now your sparring partner —
        go for a high streak.</p>""",
        setup=lambda w: _tour_setup(w),
        check=lambda w, st: _tour_check(w, st)),
]


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Kinematics Lab 2D v{APP_VERSION} — FK & IK course")
        self.setWindowIcon(app_icon())
        self.resize(1380, 900)

        # ---------------- state ----------------
        self.lengths = [float(v) for v in DEF_LENS[:2]]
        self.angles = [v * D2R for v in DEF_ANGS[:2]]
        self.mode = "fk"
        self.pts = fk_positions(self.angles, self.lengths)
        self.target = np.array(self.pts[-1])
        self.trail = deque(maxlen=1000)
        self.ghosts = []
        self.needs_solve = False
        self.obstacles = []
        self.goal = None
        self.goal_tol = 6.0
        self.ch_done = False
        self.lesson_idx = 0
        self.lesson_state = {}
        self.pins = []
        self.last_multistart = False
        self.null_on = False
        self.null_t = 0.0
        self.hold = np.zeros(2)
        self.trace_t = 0.0
        self.trace_guide = []
        self.demo_A = self.demo_B = None
        self.demo_eeA = self.demo_eeB = None
        self.demo_ghost = None
        self.demo_play = False
        self.demo_t = 0.0
        self.demo_dir = 1
        self.demo_bounces = 0
        self.demo_trail = deque(maxlen=800)
        self._demo_prev = None
        self._poll_i = 0
        self._undo_stack = []
        self.quiz = None
        self.quiz_locked = False
        self.show_grid = self.show_ws = True
        self.show_trail = self.show_angles = self.show_labels = True
        self.show_heat = self.show_manip = True
        s = QSettings("KinLab2D", "progress")
        self.done_set = set()
        for k in s.allKeys():
            if k.startswith("done/"):
                try:
                    self.done_set.add(int(k.split("/")[1]))
                except ValueError:
                    pass

        # ---------------- layout ----------------
        self.canvas = ArmCanvas(self)
        self.chart = ErrorChart(self)
        self.angle_plot = AnglePlot(self)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        lv.addWidget(self.canvas, 1)
        lv.addWidget(self.chart)
        lv.addWidget(self.angle_plot)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(400)
        self.tabs.setMaximumWidth(480)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([920, 440])
        self.setCentralWidget(self.splitter)
        self.statusBar().addWidget(QLabel(
            "  drag joints/target · wheel zoom · middle-drag pan · dbl-click "
            "reset view · right-click = obstacles · Shift = snap · Ctrl+Z undo"))
        self.progress_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.progress_lbl)

        self._build_lessons_tab()
        self._build_lab_tab()
        self._build_quiz_tab()
        self._build_theory_tab()
        self._build_menus()
        self._wire()
        self._shortcuts()

        # build solver-dependent UI state BEFORE restoring preferences,
        # so restored values can safely fire signals
        self.rebuild_fk_ui()
        self.rebuild_len_ui()
        self.rebuild_main_solver()
        self._restore_prefs()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

        last = int(QSettings("KinLab2D", "progress").value("lastlesson", 0))
        self.select_lesson(max(0, min(last, len(LESSONS) - 1)))
        self.update_progress_label()

    # ------------------------------------------------------------- UI build
    def _build_lessons_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        self.lesson_list = QListWidget()
        self.lesson_list.setMaximumHeight(170)
        for i, les in enumerate(LESSONS):
            mark = "  ✓" if i in self.done_set else ""
            self.lesson_list.addItem(f"{i}. {les['title']}{mark}")
        v.addWidget(QLabel(f"Course — {len(LESSONS)} lessons, pick one:"))
        v.addWidget(self.lesson_list)
        self.learn_browser = QTextBrowser()
        v.addWidget(self.learn_browser, 1)
        ch = QGroupBox("Your goal")
        cv = QVBoxLayout(ch)
        self.ch_label = QLabel()
        self.ch_label.setWordWrap(True)
        self.ch_label.setStyleSheet("color:#d7dbe8;")
        cv.addWidget(self.ch_label)
        row = QHBoxLayout()
        self.btn_prev_lesson = QPushButton("◂ Prev")
        self.btn_lesson_setup = QPushButton("⟲ Load setup")
        self.btn_next_lesson = QPushButton("Next ▸")
        row.addWidget(self.btn_prev_lesson)
        row.addWidget(self.btn_lesson_setup)
        row.addStretch(1)
        row.addWidget(self.btn_next_lesson)
        cv.addLayout(row)
        v.addWidget(ch)
        self.tabs.addTab(page, "Lessons")
        self.lesson_list.currentRowChanged.connect(self.select_lesson)

    def _build_lab_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "Lab")

        mode_g = QGroupBox("Mode")
        mv = QVBoxLayout(mode_g)
        self.rb_fk = QRadioButton("Forward — angles → pose   (F)")
        self.rb_ik = QRadioButton("Inverse — target → angles   (I)")
        self.rb_fk.setChecked(True)
        self.mode_desc = QLabel(FK_DESC)
        self.mode_desc.setWordWrap(True)
        self.mode_desc.setStyleSheet("color:#98a0b3;")
        mv.addWidget(self.rb_fk)
        mv.addWidget(self.rb_ik)
        mv.addWidget(self.mode_desc)
        v.addWidget(mode_g)

        self.stack = QStackedWidget()
        fk_page, fk_v = QWidget(), QVBoxLayout()
        fk_page.setLayout(fk_v)
        self.fk_box = QVBoxLayout()
        fk_v.addLayout(self.fk_box)
        row = QHBoxLayout()
        self.rand_pose_btn = QPushButton("🎲 Random pose")
        self.fk_reset_btn = QPushButton("↺ Reset pose")
        self.mirror_fk_btn = QPushButton("⇄ Mirror")
        self.mirror_fk_btn.setToolTip("negate all angles — mirror solution")
        for b in (self.rand_pose_btn, self.fk_reset_btn, self.mirror_fk_btn):
            row.addWidget(b)
        fk_v.addLayout(row)
        tip = QLabel("Grab joints on canvas · Shift+drag snaps 15° · "
                     "Ctrl+Z undoes.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#98a0b3;")
        fk_v.addWidget(tip)
        self.stack.addWidget(fk_page)

        ik_page, ik_v = QWidget(), QVBoxLayout()
        ik_page.setLayout(ik_v)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Solver"))
        self.solver_combo = QComboBox()
        for name, _f, _c in SOLVER_DEFS:
            self.solver_combo.addItem(name)
        srow.addWidget(self.solver_combo, 1)
        ik_v.addLayout(srow)
        erow = QHBoxLayout()
        self.elbow_cb = QCheckBox("Elbow up")
        self.elbow_cb.setChecked(True)
        erow.addWidget(self.elbow_cb)
        erow.addStretch(1)
        erow.addWidget(QLabel("tolerance"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.5, 30.0)
        self.tol_spin.setSingleStep(0.5)
        self.tol_spin.setValue(2.0)
        self.tol_spin.setSuffix(" u")
        erow.addWidget(self.tol_spin)
        ik_v.addLayout(erow)
        lrow = QHBoxLayout()
        self.live_cb = QCheckBox("Live follow")
        self.live_cb.setChecked(True)
        lrow.addWidget(self.live_cb)
        lrow.addStretch(1)
        lrow.addWidget(QLabel("speed"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 8)
        self.speed_slider.setValue(3)
        self.speed_slider.setMaximumWidth(90)
        self.speed_val = QLabel("×3")
        lrow.addWidget(self.speed_slider)
        lrow.addWidget(self.speed_val)
        ik_v.addLayout(lrow)
        lamrow = QHBoxLayout()
        lamrow.addWidget(QLabel("DLS λ (damping)"))
        self.lam_spin = QDoubleSpinBox()
        self.lam_spin.setRange(0.0, 3.0)
        self.lam_spin.setSingleStep(0.1)
        self.lam_spin.setValue(0.8)
        lamrow.addStretch(1)
        lamrow.addWidget(self.lam_spin)
        ik_v.addLayout(lamrow)
        brow = QHBoxLayout()
        self.step_btn = QPushButton("Step ▸  (S)")
        self.solve_btn = QPushButton("Solve ▸▸  (Space)")
        brow.addWidget(self.step_btn)
        brow.addWidget(self.solve_btn)
        ik_v.addLayout(brow)
        brow2 = QHBoxLayout()
        self.rand_t_btn = QPushButton("⟲ Random target")
        self.far_t_btn = QPushButton("✖ Out-of-reach")
        self.mirror_ik_btn = QPushButton("⇄ Swap elbow")
        for b in (self.rand_t_btn, self.far_t_btn, self.mirror_ik_btn):
            brow2.addWidget(b)
        ik_v.addLayout(brow2)

        trow = QHBoxLayout()
        self.trace_cb = QCheckBox("Trace path:")
        self.trace_kind = QComboBox()
        self.trace_kind.addItems(["Circle", "Line", "Lissajous", "Square",
                                  "Star"])
        trow.addWidget(self.trace_cb)
        trow.addWidget(self.trace_kind, 1)
        ik_v.addLayout(trow)

        nrow = QHBoxLayout()
        self.null_btn = QPushButton("Null-space ▶")
        self.null_btn.setToolTip("move joints through the null space while "
                                 "the end effector stays put (3+ links)")
        self.null_btn.setEnabled(False)
        self.null_mode = QComboBox()
        self.null_mode.addItems(["jiggle", "center joints", "max manip."])
        self.null_mode.setToolTip("what to optimise in the null space")
        nrow.addWidget(self.null_btn)
        nrow.addWidget(self.null_mode, 1)
        ik_v.addLayout(nrow)

        self.ik_status = QLabel("—")
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(10)
        self.ik_status.setFont(mono)
        self.ik_status.setStyleSheet("color:#98a0b3;")
        ik_v.addWidget(self.ik_status)
        brow3 = QHBoxLayout()
        self.ik_reset_btn = QPushButton("↺ Reset pose")
        self.compare_cb = QCheckBox("Compare all solvers")
        brow3.addWidget(self.ik_reset_btn)
        brow3.addWidget(self.compare_cb)
        ik_v.addLayout(brow3)

        og = QGroupBox("Obstacles")
        ogv = QHBoxLayout(og)
        self.obs_target_btn = QPushButton("＋ at target")
        self.obs_rand_btn = QPushButton("🎲 random")
        self.obs_clear_btn = QPushButton("clear")
        for b in (self.obs_target_btn, self.obs_rand_btn, self.obs_clear_btn):
            ogv.addWidget(b)
        self.avoid_cb = QCheckBox("avoidance")
        self.avoid_cb.setChecked(True)
        ogv.addWidget(self.avoid_cb)
        ik_v.addWidget(og)
        self.stack.addWidget(ik_page)
        v.addWidget(self.stack)

        # ----- demos & tools -----
        tools = QGroupBox("Demos & tools")
        tg = QGridLayout(tools)
        tg.setHorizontalSpacing(6)
        self.btn_pin = QPushButton("📌 Pin pose")
        self.btn_pin.setToolTip("freeze the current arm as a grey reference "
                                "ghost — compare IK branches!")
        self.btn_pins_clear = QPushButton("clear pins")
        self.both_cb = QCheckBox("show 2nd elbow")
        self.both_cb.setToolTip("2-link only: overlay the mirrored IK branch")
        tg.addWidget(self.btn_pin, 0, 0)
        tg.addWidget(self.btn_pins_clear, 0, 1)
        tg.addWidget(self.both_cb, 0, 2)
        self.btn_demo_a = QPushButton("Set A")
        self.btn_demo_a.setToolTip("capture current pose as interpolation "
                                   "start")
        self.btn_demo_b = QPushButton("Set B")
        self.btn_demo_play = QPushButton("▶ Play")
        self.btn_demo_play.setEnabled(False)
        self.btn_demo_play.setToolTip("animate: violet arm = joint-space "
                                      "lerp, main arm = task-space lerp")
        tg.addWidget(self.btn_demo_a, 1, 0)
        tg.addWidget(self.btn_demo_b, 1, 1)
        tg.addWidget(self.btn_demo_play, 1, 2)
        self.multi_btn = QPushButton("Multi-start ×24")
        self.multi_btn.setToolTip("solve from 24 random seeds, keep the best "
                                  "— the honest fix for local minima")
        tg.addWidget(self.multi_btn, 2, 0, 1, 3)
        v.addWidget(tools)

        struct = QGroupBox("Arm structure")
        sv = QVBoxLayout(struct)
        nrow2 = QHBoxLayout()
        nrow2.addWidget(QLabel("Number of links"))
        self.links_spin = QSpinBox()
        self.links_spin.setRange(1, 6)
        self.links_spin.setValue(2)
        nrow2.addStretch(1)
        nrow2.addWidget(self.links_spin)
        sv.addLayout(nrow2)
        self.len_box = QVBoxLayout()
        sv.addLayout(self.len_box)
        limrow = QHBoxLayout()
        self.limits_cb = QCheckBox("Joint limits ±")
        self.limits_spin = QSpinBox()
        self.limits_spin.setRange(10, 180)
        self.limits_spin.setValue(120)
        self.limits_spin.setSuffix("°")
        self.limits_spin.setEnabled(False)
        self.limits_cb.toggled.connect(self.limits_spin.setEnabled)
        limrow.addWidget(self.limits_cb)
        limrow.addStretch(1)
        limrow.addWidget(self.limits_spin)
        sv.addLayout(limrow)
        v.addWidget(struct)

        disp = QGroupBox("Display")
        dg = QGridLayout(disp)
        self.cb_grid = QCheckBox("Grid")
        self.cb_ws = QCheckBox("Workspace")
        self.cb_trail = QCheckBox("EE trail")
        self.cb_angles = QCheckBox("Angle arcs")
        self.cb_labels = QCheckBox("Labels")
        self.cb_heat = QCheckBox("Density map")
        self.cb_manip = QCheckBox("Manip. ellipse")
        for i, cb in enumerate((self.cb_grid, self.cb_ws, self.cb_trail,
                                self.cb_angles, self.cb_labels, self.cb_heat,
                                self.cb_manip)):
            cb.setChecked(True)
            dg.addWidget(cb, i // 2, i % 2)
        self.cb_plot = QCheckBox("Error plot")
        self.cb_plot.setChecked(True)
        dg.addWidget(self.cb_plot, 3, 0)
        self.trail_clear_btn = QPushButton("Clear trail")
        dg.addWidget(self.trail_clear_btn, 3, 1)
        self.cb_aplot = QCheckBox("Angle plot")
        self.cb_aplot.setChecked(True)
        dg.addWidget(self.cb_aplot, 4, 0)
        v.addWidget(disp)

        tele = QGroupBox("Telemetry")
        tgrid = QGridLayout(tele)
        tgrid.setVerticalSpacing(4)
        mono2 = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono2.setPointSize(10)
        self.st_ee, self.st_tgt = QLabel(), QLabel()
        self.st_err, self.st_status = QLabel(), QLabel()
        self.st_theta, self.st_reach = QLabel(), QLabel()
        self.st_manip, self.st_jac = QLabel(), QLabel()
        self.st_theta.setWordWrap(True)
        rows = [("end effector", self.st_ee), ("target", self.st_tgt),
                ("error / iters", self.st_err), ("status", self.st_status),
                ("θ relative", self.st_theta), ("reach", self.st_reach),
                ("manipulability", self.st_manip), ("Jacobian J", self.st_jac)]
        for r, (name, lbl) in enumerate(rows):
            cap = QLabel(name)
            cap.setStyleSheet("color:#98a0b3;")
            tgrid.addWidget(cap, r, 0)
            lbl.setFont(mono2)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            tgrid.addWidget(lbl, r, 1)
        v.addWidget(tele)
        v.addStretch(1)

    def _build_quiz_tab(self):
        page = QWidget()
        self.quiz_page = page
        v = QVBoxLayout(page)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)
        head = QLabel("<b>Kinematics quiz</b> — endless auto-generated "
                      "questions. Numeric FK questions can be visualised on "
                      "the canvas afterwards. Press <b>Q</b> for a new one.")
        head.setWordWrap(True)
        head.setStyleSheet("color:#98a0b3;")
        v.addWidget(head)
        self.quiz_q = QLabel()
        self.quiz_q.setWordWrap(True)
        self.quiz_q.setStyleSheet("font-size:12pt;")
        v.addWidget(self.quiz_q)
        grid = QGridLayout()
        grid.setSpacing(6)
        self.quiz_btns = []
        for i in range(4):
            b = QPushButton()
            b.setMinimumHeight(34)
            b.clicked.connect(lambda _=False, idx=i: self._on_quiz_answer(idx))
            grid.addWidget(b, i // 2, i % 2)
            self.quiz_btns.append(b)
        v.addLayout(grid)
        self.quiz_fb = QLabel()
        self.quiz_fb.setWordWrap(True)
        v.addWidget(self.quiz_fb)
        self.quiz_score = QLabel()
        v.addWidget(self.quiz_score)
        row = QHBoxLayout()
        self.quiz_show_btn = QPushButton("Show pose on canvas")
        self.quiz_show_btn.setEnabled(False)
        self.quiz_next_btn = QPushButton("Next question ▸  (Q)")
        row.addWidget(self.quiz_show_btn)
        row.addStretch(1)
        row.addWidget(self.quiz_next_btn)
        v.addLayout(row)
        v.addStretch(1)
        self.tabs.addTab(page, "Quiz")
        self.quiz_show_btn.clicked.connect(self._quiz_show_pose)
        self.quiz_next_btn.clicked.connect(self._quiz_new)
        s = QSettings("KinLab2D", "progress")
        self.quiz_correct = int(s.value("quizc", 0))
        self.quiz_total = int(s.value("quizt", 0))
        self.quiz_streak = 0
        self.quiz_best = int(s.value("quizbest", 0))
        self._quiz_new()

    def _build_theory_tab(self):
        tb = QTextBrowser()
        tb.setHtml(f"""
        <h2>Formulas cheat-sheet</h2>
        <h3>Forward kinematics</h3>
        <p>With θ<sub>i</sub> relative to the previous link, the absolute
        direction of link <i>i</i> is Θ<sub>i</sub> = θ₁+…+θ<sub>i</sub>:</p>
        <p><code>x = Σ lᵢ·cos Θᵢ&nbsp;&nbsp;&nbsp;y = Σ lᵢ·sin Θᵢ</code></p>
        <h3>Jacobian</h3>
        <p><code>J[:,i] = ( −(y<sub>e</sub>−y<sub>i</sub>) ,
        (x<sub>e</sub>−x<sub>i</sub>) )ᵀ&nbsp;&nbsp;⇒&nbsp;&nbsp;ẋ = J·θ̇</code></p>
        <h3>Inverse kinematics</h3>
        <ul>
        <li><b>Analytic</b>: <code>cos θ₂ = (x²+y²−l₁²−l₂²)/(2l₁l₂)</code>.</li>
        <li><b>Pseudo-inverse</b>: Δθ = J⁺e — fast, jittery at singularities.</li>
        <li><b>DLS</b>: Δθ = Jᵀ(JJᵀ+λ²I)⁻¹e — λ trades speed for stability.</li>
        <li><b>Transpose</b>: Δθ = α·Jᵀe — robust spring pull.</li>
        <li><b>CCD</b>: rotate each joint tip→base to aim at the target.</li>
        <li><b>FABRIK</b>: pull tip, re-fit lengths backward, then forward.</li>
        <li><b>Multi-start</b>: many random seeds → best result; beats local
        minima.</li>
        </ul>
        <h3>Manipulability &amp; singularities</h3>
        <p><code>w = √det(JJᵀ) = σ₁σ₂</code>; w = 0 at a singularity.</p>
        <h3>Redundancy &amp; null space</h3>
        <p><code>Δθ = (I − J⁺J)·z</code> moves the pose without moving the
        end effector. Objectives: jiggle, joint-centering (−θ), or gradient
        ascent on manipulability ∇w.</p>
        <h3>Interpolation</h3>
        <p>Joint-space: θ(t) = θ_A + t·(θ_B − θ_A) → curved EE path.
        Task-space: p(t) = p_A + t·(p_B − p_A) + IK per frame → straight EE
        path, risky near singularities.</p>
        <h3>Obstacles &amp; limits</h3>
        <p>Reactive relaxation: project joints out of obstacles / into limits
        after each step; the chain re-fits. Planners (RRT) handle global
        structure; IK handles the local one.</p>
        <h2>Keyboard &amp; mouse</h2>
        <table cellpadding="4">
        <tr><td><b>F / I</b></td><td>FK / IK mode</td>
            <td><b>Space</b></td><td>solve</td></tr>
        <tr><td><b>S</b></td><td>one solver step</td>
            <td><b>R</b></td><td>random target</td></tr>
        <tr><td><b>P</b></td><td>random pose</td>
            <td><b>X</b></td><td>mirror pose</td></tr>
        <tr><td><b>C</b></td><td>compare solvers</td>
            <td><b>T</b></td><td>trail</td></tr>
        <tr><td><b>G / W</b></td><td>grid / workspace</td>
            <td><b>H / E</b></td><td>heatmap / ellipse</td></tr>
        <tr><td><b>O</b></td><td>random obstacle</td>
            <td><b>N</b></td><td>next lesson</td></tr>
        <tr><td><b>Q</b></td><td>new quiz question</td>
            <td><b>Ctrl+Z</b></td><td>undo pose</td></tr>
        <tr><td><b>wheel</b></td><td>zoom</td>
            <td><b>middle-drag</b></td><td>pan canvas</td></tr>
        <tr><td><b>dbl-click</b></td><td>reset view</td>
            <td><b>Shift+drag</b></td><td>snap 15° / 10 u grid</td></tr>
        </table>
        <p><i>Version {APP_VERSION}</i></p>
        """)
        self.tabs.addTab(tb, "Theory")

    def _build_menus(self):
        def act(menu, text, slot, shortcut=None):
            a = menu.addAction(text)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            return a

        mfile = self.menuBar().addMenu("&File")
        act(mfile, "Save scene…", self.save_scene, "Ctrl+S")
        act(mfile, "Load scene…", self.load_scene, "Ctrl+O")
        act(mfile, "Export canvas PNG…", self.export_png, "Ctrl+E")
        act(mfile, "Export error log CSV…", self.export_error_csv)
        mfile.addSeparator()
        act(mfile, "Quit", self.close, "Ctrl+Q")
        mtool = self.menuBar().addMenu("&Tools")
        act(mtool, "Benchmark solvers…", self.run_benchmark)
        act(mtool, "Multi-start solve ×24", self.on_multistart)
        act(mtool, "Clear pinned poses", self.clear_pins)
        mles = self.menuBar().addMenu("&Lesson")
        act(mles, "Next lesson", self.next_lesson)
        act(mles, "Previous lesson", self.prev_lesson)
        mles.addSeparator()
        act(mles, "Reset lesson progress", self.reset_progress)
        act(mles, "Reset quiz score", self.reset_quiz_score)
        mhelp = self.menuBar().addMenu("&Help")
        act(mhelp, "Shortcuts & theory", lambda: self._goto_tab("Theory"))
        act(mhelp, "About", self.about)

    # ------------------------------------------------------------- wiring
    def _wire(self):
        self.rb_fk.toggled.connect(lambda on: on and self.set_mode("fk"))
        self.rb_ik.toggled.connect(lambda on: on and self.set_mode("ik"))
        self.solver_combo.currentIndexChanged.connect(
            lambda _: self.on_solver_changed())
        self.elbow_cb.toggled.connect(self.on_elbow)
        self.tol_spin.valueChanged.connect(self.on_tol_changed)
        self.lam_spin.valueChanged.connect(self.on_lam)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_val.setText(f"×{v}"))
        self.step_btn.clicked.connect(self.on_step)
        self.solve_btn.clicked.connect(self.on_solve)
        self.rand_t_btn.clicked.connect(self.on_random_target)
        self.far_t_btn.clicked.connect(self.on_far_target)
        self.ik_reset_btn.clicked.connect(self.on_reset_pose)
        self.rand_pose_btn.clicked.connect(self.on_random_pose)
        self.fk_reset_btn.clicked.connect(self.on_reset_pose)
        self.mirror_fk_btn.clicked.connect(self.on_mirror)
        self.mirror_ik_btn.clicked.connect(self.on_mirror)
        self.links_spin.valueChanged.connect(self.on_links_changed)
        self.compare_cb.toggled.connect(self.on_compare)
        self.avoid_cb.toggled.connect(
            lambda _o: (self.sync_solver_env(), self.refresh()))
        self.limits_cb.toggled.connect(
            lambda _o: (self.sync_solver_env(), self.refresh()))
        self.limits_spin.valueChanged.connect(
            lambda _v: (self.sync_solver_env(), self.refresh()))
        self.obs_target_btn.clicked.connect(self.add_obstacle_at_target)
        self.obs_rand_btn.clicked.connect(self.add_random_obstacle)
        self.obs_clear_btn.clicked.connect(self.clear_obstacles)
        self.trace_cb.toggled.connect(self.on_trace)
        self.trace_kind.currentIndexChanged.connect(
            lambda _i: self.on_trace_kind())
        self.null_btn.clicked.connect(self.on_null)
        self.btn_pin.clicked.connect(self.pin_pose)
        self.btn_pins_clear.clicked.connect(self.clear_pins)
        self.both_cb.toggled.connect(lambda _o: self.canvas.update())
        self.btn_demo_a.clicked.connect(self.set_demo_a)
        self.btn_demo_b.clicked.connect(self.set_demo_b)
        self.btn_demo_play.clicked.connect(self.on_demo_play)
        self.multi_btn.clicked.connect(self.on_multistart)
        self.btn_lesson_setup.clicked.connect(
            lambda: LESSONS[self.lesson_idx]["setup"](self))
        self.btn_next_lesson.clicked.connect(self.next_lesson)
        self.btn_prev_lesson.clicked.connect(self.prev_lesson)
        self.trail_clear_btn.clicked.connect(self.clear_trail)
        self.cb_plot.toggled.connect(self.chart.setVisible)
        self.cb_aplot.toggled.connect(self.angle_plot.setVisible)
        for name in ("grid", "ws", "trail", "angles", "labels", "heat",
                     "manip"):
            cb = getattr(self, f"cb_{name}")
            cb.toggled.connect(lambda on, n=name: self._display_flag(n, on))

    def _shortcuts(self):
        def sc(key, fn):
            QShortcut(QKeySequence(key), self).activated.connect(fn)
        sc("F", lambda: self.rb_fk.setChecked(True))
        sc("I", lambda: self.rb_ik.setChecked(True))
        sc("Space", self.on_solve)
        sc("S", self.on_step)
        sc("R", self.on_random_target)
        sc("P", self.on_random_pose)
        sc("X", self.on_mirror)
        sc("C", lambda: self.compare_cb.toggle())
        sc("T", lambda: self.cb_trail.toggle())
        sc("G", lambda: self.cb_grid.toggle())
        sc("W", lambda: self.cb_ws.toggle())
        sc("H", lambda: self.cb_heat.toggle())
        sc("E", lambda: self.cb_manip.toggle())
        sc("O", self.add_random_obstacle)
        sc("N", self.next_lesson)
        sc("Q", self._goto_quiz)
        sc("Ctrl+Z", self.undo)

    # ------------------------------------------------------------- helpers
    def point_reachable(self, x, y):
        if math.hypot(x, y) > sum(self.lengths) + 1e-9:
            return False
        for (ox, oy, r) in self.obstacles:
            if math.hypot(x - ox, y - oy) < r:
                return False
        return True

    def err_main(self):
        return math.hypot(self.pts[-1][0] - self.target[0],
                          self.pts[-1][1] - self.target[1])

    def manip(self):
        J = jacobian_of(self.pts, len(self.lengths))
        try:
            return float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        except Exception:
            return 0.0

    def joint_limits(self):
        if self.limits_cb.isChecked():
            d = self.limits_spin.value() * D2R
        else:
            d = math.pi
        return [(-d, d)] * len(self.lengths)

    def clamp_angles(self):
        if not self.limits_cb.isChecked():
            return
        d = self.limits_spin.value() * D2R
        for i in range(len(self.angles)):
            self.angles[i] = max(-d, min(d, wrap(self.angles[i])))

    def _display_flag(self, name, on):
        setattr(self, f"show_{name}", bool(on))
        self.canvas.update()

    def clear_trail(self):
        self.trail.clear()
        self.canvas.update()

    def update_progress_label(self):
        self.progress_lbl.setText(
            f"lessons done: {len(self.done_set)}/{len(LESSONS)}  ")

    def push_undo(self):
        self._undo_stack.append(list(self.angles))
        if len(self._undo_stack) > 60:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            self.statusBar().showMessage("nothing to undo", 1500)
            return
        self.angles = self._undo_stack.pop()
        self.clamp_angles()
        self.refresh()

    def _goto_tab(self, name):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name:
                self.tabs.setCurrentIndex(i)
                return

    def about(self):
        QMessageBox.about(self, "About Kinematics Lab",
                          f"<b>Kinematics Lab 2D</b> v{APP_VERSION}<br>"
                          "An interactive course for 2D forward &amp; inverse "
                          f"kinematics: {len(LESSONS)} lessons, quiz, 6 "
                          "solvers, obstacles, redundancy, benchmarks.<br><br>"
                          "Built with PySide6 + numpy.")

    # ------------------------------------------------------- preferences
    _PREF_TOGGLES = ("grid", "ws", "trail", "angles", "labels", "heat",
                     "manip")

    @staticmethod
    def _truthy(v):
        return v in (True, "true", "1", 1)

    def _save_prefs(self):
        s = QSettings("KinLab2D", "prefs")
        s.setValue("geom", self.saveGeometry())
        s.setValue("split", self.splitter.saveState())
        for n in self._PREF_TOGGLES:
            s.setValue(f"disp/{n}", getattr(self, f"cb_{n}").isChecked())
        s.setValue("disp/plot", self.cb_plot.isChecked())
        s.setValue("disp/aplot", self.cb_aplot.isChecked())
        s.setValue("solver", self.solver_combo.currentIndex())
        s.setValue("tol", self.tol_spin.value())
        s.setValue("lam", self.lam_spin.value())
        s.setValue("live", self.live_cb.isChecked())
        s.setValue("speed", self.speed_slider.value())
        s.setValue("avoid", self.avoid_cb.isChecked())
        s.setValue("limits_on", self.limits_cb.isChecked())
        s.setValue("limits_deg", self.limits_spin.value())
        s.setValue("tab", self.tabs.currentIndex())

    def _restore_prefs(self):
        try:
            s = QSettings("KinLab2D", "prefs")
            g = s.value("geom")
            if g is not None:
                self.restoreGeometry(g)
            sp = s.value("split")
            if sp is not None:
                self.splitter.restoreState(sp)
            for n in self._PREF_TOGGLES:
                v = s.value(f"disp/{n}")
                if v is not None:
                    getattr(self, f"cb_{n}").setChecked(self._truthy(v))
            for key, cb in (("disp/plot", self.cb_plot),
                            ("disp/aplot", self.cb_aplot)):
                v = s.value(key)
                if v is not None:
                    cb.setChecked(self._truthy(v))
            v = s.value("solver")
            if v is not None:
                self.solver_combo.setCurrentIndex(int(v))
            for key, spin in (("tol", self.tol_spin), ("lam", self.lam_spin),
                              ("speed", self.speed_slider),
                              ("limits_deg", self.limits_spin)):
                v = s.value(key)
                if v is not None:
                    spin.setValue(float(v))
            for key, cb in (("live", self.live_cb), ("avoid", self.avoid_cb),
                            ("limits_on", self.limits_cb)):
                v = s.value(key)
                if v is not None:
                    cb.setChecked(self._truthy(v))
            v = s.value("tab")
            if v is not None:
                self.tabs.setCurrentIndex(int(v))
        except Exception:
            pass

    def closeEvent(self, e):
        self._save_prefs()
        super().closeEvent(e)

    # ------------------------------------------------------------- solvers
    def _env_solver(self, s):
        s.obstacles = list(self.obstacles)
        s.avoid = self.avoid_cb.isChecked()
        s.use_limits = self.limits_cb.isChecked()
        s.limits = self.joint_limits()
        if isinstance(s, JacobianSolver):
            s.lam = self.lam_spin.value()

    def sync_solver_env(self):
        for s in [getattr(self, "solver", None)] + [g[2] for g in self.ghosts]:
            if s is not None:
                self._env_solver(s)

    def rebuild_main_solver(self):
        _name, factory, _c = SOLVER_DEFS[self.solver_combo.currentIndex()]
        self.solver = factory()
        self.solver.reset(self.angles, self.lengths, self.target,
                          self.tol_spin.value())
        if isinstance(self.solver, AnalyticSolver):
            self.solver.elbow_up = self.elbow_cb.isChecked()
        self.sync_solver_env()
        self.needs_solve = True

    def reset_ghosts(self):
        self.ghosts = []
        for name, factory, color in SOLVER_DEFS:
            if name.startswith("Analytic") and len(self.lengths) != 2:
                continue
            s = factory()
            s.reset(self.angles, self.lengths, self.target,
                    self.tol_spin.value())
            if isinstance(s, AnalyticSolver):
                s.elbow_up = self.elbow_cb.isChecked()
            self.ghosts.append((name, QColor(color), s))
        self.sync_solver_env()

    def set_target(self, xy):
        self.target = np.array(xy, float)
        if self.mode == "ik":
            self.solver.reset(self.angles, self.lengths, self.target,
                              self.tol_spin.value())
            if self.compare_cb.isChecked():
                self.reset_ghosts()
            self.needs_solve = True
            self.chart.clear()
        self.refresh()

    def move_target(self, xy):
        """Retarget without resetting solvers (used by trace mode)."""
        self.target = np.array(xy, float)
        if self.mode == "ik":
            self.solver.target = self.target
            if self.compare_cb.isChecked():
                for _n, _c, s in self.ghosts:
                    s.target = self.target
            self.needs_solve = True
        self.refresh()

    # ------------------------------------------------------------- handlers
    def on_fk_slider(self, i, v):
        self.angles[i] = v * D2R
        self.refresh()

    def on_len_slider(self, i, v):
        self.lengths[i] = float(v)
        self.len_sliders[i][1].setText(f"{v} u")
        self.trail.clear()
        self.canvas.invalidate_ws()
        self.update_trace_guide()
        self.rebuild_main_solver()
        if self.compare_cb.isChecked():
            self.reset_ghosts()
        self.refresh()

    def on_links_changed(self, n):
        while len(self.lengths) < n:
            self.lengths.append(float(DEF_LENS[len(self.lengths)]))
        del self.lengths[n:]
        while len(self.angles) < n:
            self.angles.append(0.0)
        del self.angles[n:]
        self.rebuild_fk_ui()
        self.rebuild_len_ui()
        self.canvas.invalidate_ws()
        self.update_trace_guide()
        self.trail.clear()
        self.angle_plot.clear()
        self.both_cb.setEnabled(n == 2)
        self.solver_combo.model().item(0).setEnabled(n == 2)
        if n != 2 and self.solver_combo.currentIndex() == 0:
            self.solver_combo.setCurrentIndex(1)
        else:
            self.rebuild_main_solver()
            if self.compare_cb.isChecked():
                self.reset_ghosts()
        self.null_btn.setEnabled(n >= 3)
        self.null_mode.setEnabled(n >= 3)
        if n < 3 and self.null_on:
            self.null_on = False
            self.null_btn.setText("Null-space ▶")
        if self.demo_play:
            self.on_demo_play()
        self.refresh()

    def on_solver_changed(self):
        is_analytic = SOLVER_DEFS[self.solver_combo.currentIndex()][0].startswith(
            "Analytic")
        self.elbow_cb.setEnabled(is_analytic)
        self.rebuild_main_solver()
        if self.compare_cb.isChecked():
            self.reset_ghosts()
        self.refresh()

    def on_elbow(self, on):
        for s in [self.solver] + [g[2] for g in self.ghosts]:
            if isinstance(s, AnalyticSolver):
                s.elbow_up = bool(on)
        self.needs_solve = True
        self.refresh()

    def on_tol_changed(self, v):
        self.solver.tolerance = v
        for _n, _c, s in self.ghosts:
            s.tolerance = v
        self.needs_solve = True
        self.refresh()

    def on_lam(self, v):
        for s in [self.solver] + [g[2] for g in self.ghosts]:
            if isinstance(s, JacobianSolver):
                s.lam = float(v)
        self.needs_solve = True

    def on_compare(self, on):
        if on:
            self.reset_ghosts()
        self.refresh()

    def on_step(self):
        if self.mode != "ik":
            return
        self.chart.add(self.solver.step())
        if self.compare_cb.isChecked():
            for _n, _c, s in self.ghosts:
                if s.error() > self.tol_spin.value() and s.iterations < s.max_iter:
                    s.step()
        self.angles = list(self.solver.angles)
        self.refresh()

    def on_solve(self):
        if self.mode != "ik":
            return
        err = 1e9
        while (self.solver.iterations < self.solver.max_iter
               and err > self.tol_spin.value()):
            err = self.solver.step()
            self.chart.add(err)
        self.angles = list(self.solver.angles)
        self.needs_solve = err > self.tol_spin.value()
        self.refresh()

    def on_random_target(self):
        total = sum(self.lengths)
        r = total * math.sqrt(np.random.random()) * 0.96
        a = np.random.random() * 2 * math.pi
        self.set_target((r * math.cos(a), r * math.sin(a)))

    def on_far_target(self):
        total = sum(self.lengths)
        a = math.radians(25)
        self.set_target((1.3 * total * math.cos(a), 1.3 * total * math.sin(a)))

    def on_random_pose(self):
        self.push_undo()
        self.angles = [np.random.uniform(-160, 160) * D2R
                       for _ in self.angles]
        self.refresh()

    def on_mirror(self):
        self.push_undo()
        self.angles = [-a for a in self.angles]
        self.clamp_angles()
        if self.mode == "ik":
            self.rebuild_main_solver()
            if self.compare_cb.isChecked():
                self.reset_ghosts()
        self.refresh()

    def on_reset_pose(self):
        self.push_undo()
        self.angles = [v * D2R for v in DEF_ANGS[:len(self.lengths)]]
        self.trail.clear()
        self.chart.clear()
        self.angle_plot.clear()
        if self.mode == "ik":
            self.rebuild_main_solver()
            if self.compare_cb.isChecked():
                self.reset_ghosts()
        self.refresh()

    # pins
    def pin_pose(self):
        if len(self.pins) >= 5:
            self.statusBar().showMessage("pin limit reached (5)", 2500)
            return
        self.pins.append((list(self.angles), list(self.lengths)))
        self.canvas.update()
        self.statusBar().showMessage("pose pinned — compare it against new "
                                     "solutions", 3000)

    def clear_pins(self):
        self.pins.clear()
        self.canvas.update()

    # ------------------------------------------------------- multistart
    def on_multistart(self):
        if self.mode != "ik":
            self.statusBar().showMessage("switch to IK mode first", 2500)
            return
        if SOLVER_DEFS[self.solver_combo.currentIndex()][0].startswith("Analytic"):
            self.statusBar().showMessage("analytic IK is deterministic — "
                                         "multistart adds nothing here", 3500)
        tol = self.tol_spin.value()
        _n, factory, _c = SOLVER_DEFS[self.solver_combo.currentIndex()]
        best = None
        for _k in range(24):
            seed = [np.random.uniform(-math.pi, math.pi)
                    for _ in self.lengths]
            s = factory()
            s.reset(seed, self.lengths, self.target, tol)
            self._env_solver(s)
            guard = 0
            while s.error() > tol and s.iterations < 800 and guard < 820:
                s.step()
                guard += 1
            e = s.error()
            if best is None or e < best[0]:
                best = (e, s)
        e, s = best
        self.last_multistart = True
        self.solver.angles = list(s.angles)
        self.solver.iterations = s.iterations
        self.solver.ee_path = s.ee_path
        self.angles = list(s.angles)
        self.chart.add(e)
        self.statusBar().showMessage(
            f"multi-start: best error {e:.2f} u in {s.iterations} iters "
            f"(of 24 starts)", 4000)
        self.refresh()

    # ------------------------------------------------------- benchmark
    def run_benchmark(self):
        trials = 20
        tol = self.tol_spin.value()
        R = sum(self.lengths)
        rng = np.random.default_rng(42)
        rows = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            for name, factory, color in SOLVER_DEFS:
                if name.startswith("Analytic") and len(self.lengths) != 2:
                    continue
                succ, iters, paths, us = 0, [], [], []
                for _k in range(trials):
                    a = rng.random() * 2 * math.pi
                    r = R * math.sqrt(rng.random()) * 0.97
                    tgt = np.array([r * math.cos(a), r * math.sin(a)])
                    seed = [rng.uniform(-math.pi, math.pi)
                            for _ in self.lengths]
                    s = factory()
                    s.reset(seed, self.lengths, tgt, tol)
                    while s.error() > tol and s.iterations < 1500:
                        s.step()
                    us.append(s.us_per_it)
                    if s.error() <= tol:
                        succ += 1
                        iters.append(s.iterations)
                        paths.append(s.ee_path)
                rows.append((
                    name, QColor(color), succ, trials,
                    f"{statistics.median(iters):.0f}" if iters else "—",
                    f"{statistics.median(paths):.0f}" if paths else "—",
                    f"{statistics.median(us):.1f}" if us else "—",
                    statistics.median(us) if us else float("inf"),
                ))
        finally:
            QApplication.restoreOverrideCursor()

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Solver benchmark — {len(self.lengths)} links, "
                           f"tol {tol:.1f} u, {trials} random targets")
        v = QVBoxLayout(dlg)
        note = QLabel(f"Arm lengths: "
                      f"{', '.join(f'{v:.0f}' for v in self.lengths)} · "
                      "median over successful runs · pure IK (no obstacles/"
                      "limits)")
        note.setStyleSheet("color:#98a0b3;")
        v.addWidget(note)
        tw = QTableWidget(len(rows), 6)
        tw.setHorizontalHeaderLabels(["solver", "solved", "med iters",
                                      "med EE path", "µs/iter", ""])
        tw.setEditTriggers(QTableWidget.NoEditTriggers)
        tw.verticalHeader().setVisible(False)
        fastest = min(r[7] for r in rows)
        for r, (name, color, succ, tot, mi, mp, mu, us_f) in enumerate(rows):
            it_name = QTableWidgetItem(name)
            it_name.setForeground(color)
            tw.setItem(r, 0, it_name)
            tw.setItem(r, 1, QTableWidgetItem(f"{succ}/{tot}"))
            tw.setItem(r, 2, QTableWidgetItem(mi))
            tw.setItem(r, 3, QTableWidgetItem(mp))
            tw.setItem(r, 4, QTableWidgetItem(mu))
            tw.setItem(r, 5, QTableWidgetItem(
                "⚡ fastest" if us_f <= fastest + 1e-9 else ""))
        tw.resizeColumnsToContents()
        v.addWidget(tw)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        v.addWidget(btn, 0, Qt.AlignRight)
        dlg.resize(560, 320)
        dlg.exec()

    # obstacles
    def add_obstacle(self, x, y, r):
        if len(self.obstacles) >= 6:
            self.statusBar().showMessage("obstacle limit reached", 2500)
            return
        self.obstacles.append((float(x), float(y), float(r)))
        self.sync_solver_env()
        self.refresh()

    def add_obstacle_at_target(self):
        self.add_obstacle(self.target[0], self.target[1], 26.0)

    def add_random_obstacle(self):
        total = sum(self.lengths)
        a = np.random.random() * 2 * math.pi
        r = total * np.random.uniform(0.3, 0.75)
        self.add_obstacle(r * math.cos(a), r * math.sin(a),
                          float(np.random.uniform(22, 46)))

    def remove_obstacle(self, i):
        if 0 <= i < len(self.obstacles):
            self.obstacles.pop(i)
            self.sync_solver_env()
            self.refresh()

    def clear_obstacles(self):
        self.obstacles.clear()
        self.sync_solver_env()
        self.refresh()

    # trace
    def update_trace_guide(self):
        R = sum(self.lengths)
        kind = self.trace_kind.currentIndex()
        self.trace_guide = [path_point(kind, i / 240 * 2 * math.pi, R)
                            for i in range(240)]

    def on_trace(self, on):
        if on and self.mode != "ik":
            self.rb_ik.setChecked(True)
        if on:
            self.trace_t = 0.0
            self.update_trace_guide()
        self.refresh()

    def on_trace_kind(self):
        self.trace_t = 0.0
        self.update_trace_guide()
        self.refresh()

    # ------------------------------------------------------- interpolation
    def set_demo_a(self):
        self.demo_A = list(self.angles)
        self.demo_eeA = np.array(self.pts[-1])
        self.btn_demo_a.setText("A ✓")
        self._demo_ready()

    def set_demo_b(self):
        self.demo_B = list(self.angles)
        self.demo_eeB = np.array(self.pts[-1])
        self.btn_demo_b.setText("B ✓")
        self._demo_ready()

    def _demo_ready(self):
        self.btn_demo_play.setEnabled(bool(self.demo_A and self.demo_B))

    def on_demo_play(self):
        if not (self.demo_A and self.demo_B):
            self.statusBar().showMessage("set poses A and B first", 2500)
            return
        self.demo_play = not self.demo_play
        if self.demo_play:
            self.demo_t = 0.0
            self.demo_dir = 1
            self._demo_prev = None
            self.demo_trail.clear()
        self.btn_demo_play.setText("⏸ Pause" if self.demo_play else "▶ Play")

    def _demo_solve_task(self, tgt, seed):
        if len(self.lengths) == 2:
            a = analytic_angles(self.lengths, tgt,
                                self.elbow_cb.isChecked())
            return a if a is not None else list(seed)
        s = FABRIKSolver()
        s.reset(seed, self.lengths, tgt, 0.5)
        for _ in range(20):
            s.step()
        return s.angles

    def _demo_tick(self):
        if not (self.demo_A and self.demo_B):
            self.demo_play = False
            self.btn_demo_play.setText("▶ Play")
            return
        self.demo_t += 0.006 * self.speed_slider.value() * self.demo_dir
        if self.demo_t >= 1.0:
            self.demo_t = 1.0
            self.demo_dir = -1
            self.demo_bounces += 1
        elif self.demo_t <= 0.0:
            self.demo_t = 0.0
            self.demo_dir = 1
            self.demo_bounces += 1
        t = self.demo_t
        ease = t * t * (3 - 2 * t)                       # smoothstep
        qj = [a + wrap(b - a) * ease
              for a, b in zip(self.demo_A, self.demo_B)]
        tgt = (self.demo_eeA * (1 - ease) + self.demo_eeB * ease)
        seed = self._demo_prev if self._demo_prev is not None else self.demo_A
        task = self._demo_solve_task(tuple(tgt), seed)
        self._demo_prev = task
        self.demo_ghost = qj
        ee_j = fk_positions(qj, self.lengths)[-1]
        self.demo_trail.append(ee_j)
        self.angles = list(task)
        self.refresh()

    # null space
    def on_null(self):
        if len(self.lengths) < 3:
            self.statusBar().showMessage("null space needs 3+ links", 2500)
            return
        self.null_on = not self.null_on
        if self.null_on:
            self.hold = np.array(self.pts[-1])
            self.null_t = 0.0
        self.null_btn.setText("⏹ Stop" if self.null_on else "Null-space ▶")

    def _null_tick(self):
        n = len(self.lengths)
        if n < 3:
            self.null_on = False
            self.null_btn.setText("Null-space ▶")
            return
        J = jacobian_of(self.pts, n)
        ee = np.array(self.pts[-1])
        Jp = np.linalg.pinv(J)
        N = np.eye(n) - Jp @ J
        mode = self.null_mode.currentIndex()
        if mode == 0:                                   # jiggle
            pref = np.array([0.7 * math.sin(self.null_t + 1.1 * i)
                             for i in range(n)])
        elif mode == 1:                                 # center joints
            pref = -np.array([wrap(a) for a in self.angles]) * 1.4
        else:                                           # maximize manipulability
            eps = 2e-3
            w0 = self.manip()
            pref = np.zeros(n)
            for i in range(n):
                a2 = list(self.angles)
                a2[i] = wrap(a2[i] + eps)
                J2 = jacobian_of(fk_positions(a2, self.lengths), n)
                try:
                    w2 = math.sqrt(max(np.linalg.det(J2 @ J2.T), 0.0))
                except Exception:
                    w2 = w0
                pref[i] = (w2 - w0) / eps
            nn = float(np.linalg.norm(pref))
            if nn > 1e-9:
                pref = pref * (0.5 / nn)
        dq = N @ pref + Jp @ (0.9 * (self.hold - ee))
        m = float(np.abs(dq).max())
        if m > 0.05:
            dq *= 0.05 / m
        self.angles = [wrap(a + dq[i]) for i, a in enumerate(self.angles)]
        self.clamp_angles()
        self.null_t += 0.06 * self.speed_slider.value()
        self.refresh()

    def _trace_tick(self):
        R = sum(self.lengths)
        kind = self.trace_kind.currentIndex()
        self.trace_t = (self.trace_t
                        + 0.0045 * self.speed_slider.value()) % (2 * math.pi)
        self.move_target(path_point(kind, self.trace_t, R))
        speed = self.speed_slider.value()
        for _ in range(speed):
            self.chart.add(self.solver.step())
        if self.compare_cb.isChecked():
            for _n, _c, s in self.ghosts:
                for _ in range(speed):
                    if s.iterations < s.max_iter:
                        s.step()
        self.refresh()

    # ------------------------------------------------------------- main loop
    def _tick(self):
        self._poll_i = (self._poll_i + 1) % 6
        if self._poll_i == 0:
            self.poll_challenge()
        if self.demo_play:
            self._demo_tick()
            return
        if self.mode != "ik":
            if self.null_on:
                self._null_tick()
            return
        if self.trace_cb.isChecked():
            self._trace_tick()
            return
        live = self.live_cb.isChecked()
        tol = self.tol_spin.value()
        did = False
        if live and self.needs_solve:
            for _ in range(self.speed_slider.value()):
                err = self.solver.step()
                self.chart.add(err)
                if err <= tol or self.solver.iterations >= self.solver.max_iter:
                    self.needs_solve = False
                    break
            self.angles = list(self.solver.angles)
            did = True
        if live and self.compare_cb.isChecked():
            for _n, _c, s in self.ghosts:
                if s.error() > tol and s.iterations < s.max_iter:
                    for _ in range(self.speed_slider.value()):
                        s.step()
                        if s.error() <= tol or s.iterations >= s.max_iter:
                            break
            did = True
        if did:
            self.refresh()
        if self.null_on:
            self._null_tick()

    # ------------------------------------------------------------- modes
    def set_mode(self, m, force=False):
        if m == self.mode and not force:
            return
        if self.demo_play:
            self.on_demo_play()                       # pause demo on switch
        self.mode = m
        self.stack.setCurrentIndex(0 if m == "fk" else 1)
        self.mode_desc.setText(FK_DESC if m == "fk" else IK_DESC)
        self.canvas.setCursor(Qt.CrossCursor if m == "ik" else Qt.ArrowCursor)
        if m == "ik":
            self.target = np.array(self.pts[-1])
            self.rebuild_main_solver()
            if self.compare_cb.isChecked():
                self.reset_ghosts()
            self.trail.clear()
        self.refresh()

    # ------------------------------------------------------------- refresh
    def rebuild_fk_ui(self):
        while self.fk_box.count():
            it = self.fk_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.fk_sliders = []
        for i in range(len(self.angles)):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(f"θ{i+1}")
            lab.setMinimumWidth(26)
            s = QSlider(Qt.Horizontal)
            s.setRange(-180, 180)
            s.setToolTip("joint angle relative to the previous link")
            val = QLabel()
            val.setMinimumWidth(60)
            h.addWidget(lab)
            h.addWidget(s, 1)
            h.addWidget(val)
            self.fk_box.addWidget(row)
            s.valueChanged.connect(lambda v, idx=i: self.on_fk_slider(idx, v))
            s.sliderPressed.connect(self.push_undo)
            self.fk_sliders.append((s, val))
        self.sync_fk_sliders()

    def rebuild_len_ui(self):
        while self.len_box.count():
            it = self.len_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.len_sliders = []
        for i in range(len(self.lengths)):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(f"L{i+1}")
            lab.setMinimumWidth(26)
            s = QSlider(Qt.Horizontal)
            s.setRange(40, 160)
            s.setValue(int(self.lengths[i]))
            val = QLabel(f"{self.lengths[i]:.0f} u")
            val.setMinimumWidth(52)
            h.addWidget(lab)
            h.addWidget(s, 1)
            h.addWidget(val)
            self.len_box.addWidget(row)
            s.valueChanged.connect(lambda v, idx=i: self.on_len_slider(idx, v))
            self.len_sliders.append((s, val))

    def sync_fk_sliders(self):
        for i, (s, val) in enumerate(self.fk_sliders):
            s.blockSignals(True)
            s.setValue(int(round(self.angles[i] * R2D)))
            cum = sum(self.angles[:i + 1]) * R2D
            val.setText(f"{self.angles[i]*R2D:+.0f}° "
                        f"<span style='color:#8891a8'>(abs {cum:+.0f}°)</span>")
            s.blockSignals(False)

    def refresh(self):
        self.pts = fk_positions(self.angles, self.lengths)
        ee = self.pts[-1]
        if not self.trail or math.hypot(ee[0] - self.trail[-1][0],
                                        ee[1] - self.trail[-1][1]) > 1.2:
            self.trail.append((ee[0], ee[1]))
        self.angle_plot.add(self.angles)
        self.canvas.update()
        if self.mode == "fk":
            self.sync_fk_sliders()
        self.update_stats()

    def update_stats(self):
        ee = self.pts[-1]
        self.st_ee.setText(f"({ee[0]:7.1f}, {ee[1]:7.1f}) u")
        if self.mode == "ik":
            t = self.target
            self.st_tgt.setText(f"({t[0]:7.1f}, {t[1]:7.1f}) u")
            err = self.err_main()
            it = self.solver.iterations
            tol = self.tol_spin.value()
            us = self.solver.us_per_it
            self.st_err.setText(f"{err:7.2f} u · {it} it · "
                                f"path {self.solver.ee_path:.0f}")
            if err <= tol:
                st, col = "converged ✓", "#3ecf8e"
            elif it >= self.solver.max_iter:
                st, col = "stuck — unreachable?", "#f7b32b"
            else:
                st, col = "solving…", "#8fb7ff"
            self.st_status.setText(st)
            self.st_status.setStyleSheet(f"color:{col}; font-weight:600;")
            extra = f" · {us:.0f} µs/it" if it > 0 else ""
            self.ik_status.setText(f"iter {it} · err {err:.2f} u · {st}"
                                   f"{extra}")
        else:
            self.st_tgt.setText("—")
            self.st_err.setText("—")
            self.st_status.setText("FK: pose from angles")
            self.st_status.setStyleSheet("")
            self.ik_status.setText("—")
        self.st_theta.setText("  ".join(f"{a*R2D:+.1f}°" for a in self.angles))
        total = sum(self.lengths)
        rmin = (max(0.0, 2 * max(self.lengths) - total)
                if len(self.lengths) > 1 else total)
        frac = 1.0 - (rmin / total) ** 2 if total > 0 else 0.0
        self.st_reach.setText(f"{total:.0f} u max · {rmin:.0f} u min · "
                              f"{frac*100:.0f}% of disc")
        w = self.manip()
        self.st_manip.setText(f"w = {w:8.1f}")
        self.st_manip.setStyleSheet(
            "color:#ef6461; font-weight:600;" if w < 12.0 else "")
        if isinstance(getattr(self, "solver", None), JacobianSolver):
            J = jacobian_of(self.pts, len(self.lengths))
            self.st_jac.setText(
                f"[ " + "  ".join(f"{v:7.1f}" for v in J[0]) + " ]\n"
                f"[ " + "  ".join(f"{v:7.1f}" for v in J[1]) + " ]")
        else:
            self.st_jac.setText("—")

    # ------------------------------------------------------------- lessons
    def _refresh_lesson_marks(self):
        for i, les in enumerate(LESSONS):
            mark = "  ✓" if i in self.done_set else ""
            self.lesson_list.item(i).setText(f"{i}. {les['title']}{mark}")

    def _reset_demo(self):
        self.demo_play = False
        self.demo_A = self.demo_B = None
        self.demo_eeA = self.demo_eeB = None
        self.demo_ghost = None
        self.demo_t = 0.0
        self.demo_dir = 1
        self.demo_bounces = 0
        self._demo_prev = None
        self.demo_trail.clear()
        self.btn_demo_a.setText("Set A")
        self.btn_demo_b.setText("Set B")
        self.btn_demo_play.setText("▶ Play")
        self.btn_demo_play.setEnabled(False)

    def select_lesson(self, idx):
        idx = max(0, min(idx, len(LESSONS) - 1))
        self.lesson_idx = idx
        self.lesson_list.blockSignals(True)
        self.lesson_list.setCurrentRow(idx)
        self.lesson_list.blockSignals(False)
        les = LESSONS[idx]
        self.setWindowTitle(f"Kinematics Lab 2D — lesson {idx+1}/"
                            f"{len(LESSONS)}: {les['title']}")
        QSettings("KinLab2D", "progress").setValue("lastlesson", idx)
        self.learn_browser.setHtml(les["theory"])
        self.ch_done = False
        self.ch_label.setText("◎  " + les["challenge"])
        self.ch_label.setStyleSheet("color:#d7dbe8;")
        self.null_on = False
        self.null_btn.setText("Null-space ▶")
        self._reset_demo()
        self.pins.clear()
        self.last_multistart = False
        self.lesson_state = {}
        if les["setup"]:
            les["setup"](self)
        self._refresh_lesson_marks()

    def next_lesson(self):
        self.select_lesson((self.lesson_idx + 1) % len(LESSONS))

    def prev_lesson(self):
        self.select_lesson((self.lesson_idx - 1) % len(LESSONS))

    def reset_progress(self):
        s = QSettings("KinLab2D", "progress")
        for k in list(s.allKeys()):
            if k.startswith("done/"):
                s.remove(k)
        s.sync()
        self.done_set.clear()
        self._refresh_lesson_marks()
        self.update_progress_label()
        self.statusBar().showMessage("lesson progress reset", 3000)

    def reset_quiz_score(self):
        s = QSettings("KinLab2D", "progress")
        for k in ("quizc", "quizt", "quizbest"):
            s.remove(k)
        s.sync()
        self.quiz_correct = self.quiz_total = 0
        self.quiz_streak = self.quiz_best = 0
        self._quiz_score_update()
        self.statusBar().showMessage("quiz score reset", 3000)

    def poll_challenge(self):
        les = LESSONS[self.lesson_idx]
        if self.ch_done or les["check"] is None:
            return
        try:
            done, msg = les["check"](self, self.lesson_state)
        except Exception as e:
            done, msg = False, f"(checker error: {e})"
        self.ch_label.setText(("✓  " if done else "◎  ") + str(msg))
        self.ch_label.setStyleSheet(
            "color:#3ecf8e; font-weight:600;" if done else "color:#d7dbe8;")
        if done:
            self.ch_done = True
            self.done_set.add(self.lesson_idx)
            s = QSettings("KinLab2D", "progress")
            s.setValue(f"done/{self.lesson_idx}", True)
            s.sync()
            self._refresh_lesson_marks()
            self.update_progress_label()
            self.statusBar().showMessage("Lesson complete ✓", 4000)

    # ------------------------------------------------------------- setup
    def apply_setup(self, n, lengths, angles_deg, mode="ik", solver=None,
                    target=None, goal=None, obstacles=(), tol=None, lam=None,
                    limits=None, avoid=None, compare=None, trace=False,
                    trace_kind=0, live=None):
        self.links_spin.blockSignals(True)
        self.links_spin.setValue(n)
        self.links_spin.blockSignals(False)
        self.lengths = [float(v) for v in lengths]
        self.angles = [a * D2R for a in angles_deg]
        while len(self.angles) < n:
            self.angles.append(0.0)
        del self.angles[n:]
        self.pts = fk_positions(self.angles, self.lengths)
        self.rebuild_fk_ui()
        self.rebuild_len_ui()
        self.canvas.invalidate_ws()
        self.both_cb.setEnabled(n == 2)
        (self.rb_fk if mode == "fk" else self.rb_ik).setChecked(True)
        if solver is not None:
            self.solver_combo.setCurrentIndex(solver)
        self.rebuild_main_solver()
        if tol is not None:
            self.tol_spin.setValue(tol)
        if lam is not None:
            self.lam_spin.setValue(lam)
        if limits is not None:
            self.limits_cb.blockSignals(True)
            self.limits_cb.setChecked(limits > 0)
            self.limits_cb.blockSignals(False)
            self.limits_spin.blockSignals(True)
            self.limits_spin.setValue(limits if limits > 0 else 120)
            self.limits_spin.blockSignals(False)
        self.sync_solver_env()
        self.set_target(target if target is not None else self.pts[-1])
        self.goal = tuple(goal) if goal else None
        self.obstacles = [tuple(o) for o in obstacles]
        self.sync_solver_env()
        if avoid is not None:
            self.avoid_cb.setChecked(avoid)
        if compare is not None:
            self.compare_cb.setChecked(compare)
        if live is not None:
            self.live_cb.setChecked(live)
        self.trace_cb.blockSignals(True)
        self.trace_cb.setChecked(trace)
        self.trace_cb.blockSignals(False)
        self.trace_kind.setCurrentIndex(trace_kind)
        self.trace_t = 0.0
        self.update_trace_guide()
        self.trail.clear()
        self.chart.clear()
        self.angle_plot.clear()
        self.needs_solve = True
        self.refresh()

    # ------------------------------------------------------------- quiz
    def _quiz_score_update(self):
        self.quiz_score.setText(
            f"Score: {self.quiz_correct}/{self.quiz_total} · streak "
            f"{self.quiz_streak} (best {self.quiz_best})  · saved")

    def _quiz_new(self):
        rng = np.random.default_rng()
        r = rng.random()
        if r < 0.40:
            q = quiz_gen_fk(rng)
        elif r < 0.55:
            q = quiz_gen_reach(rng)
        elif r < 0.70:
            q = quiz_gen_maxreach(rng)
        elif r < 0.80:
            q = quiz_gen_absang(rng)
        else:
            item = QUIZ_BANK[int(rng.integers(0, len(QUIZ_BANK)))]
            q = {"q": item[0], "options": list(item[1]),
                 "correct": item[2], "explain": item[3]}
        perm = rng.permutation(len(q["options"]))
        q["options"] = [q["options"][i] for i in perm]
        q["correct"] = int(list(perm).index(q["correct"]))
        self.quiz = q
        self.quiz_locked = False
        self.quiz_q.setText("<b>" + q["q"] + "</b>")
        for i, b in enumerate(self.quiz_btns):
            if i < len(q["options"]):
                b.setEnabled(True)
                b.setText(q["options"][i])
            else:
                b.setEnabled(False)
                b.setText("—")
            b.setStyleSheet("")
        self.quiz_fb.setText("")
        self.quiz_show_btn.setEnabled(bool(q.get("show")))
        self._quiz_score_update()

    def _on_quiz_answer(self, i):
        if self.quiz_locked or not self.quiz:
            return
        self.quiz_locked = True
        c = self.quiz["correct"]
        self.quiz_total += 1
        ok_green = "QPushButton{background:#2e6b4f; border-color:#3ecf8e;}"
        bad_red = "QPushButton{background:#6b2e2e; border-color:#ef6461;}"
        if i == c:
            self.quiz_correct += 1
            self.quiz_streak += 1
            self.quiz_best = max(self.quiz_best, self.quiz_streak)
            self.quiz_fb.setText("✓ <b>Correct!</b>  " + self.quiz["explain"])
            self.quiz_fb.setStyleSheet("color:#3ecf8e;")
            self.quiz_btns[i].setStyleSheet(ok_green)
        else:
            self.quiz_streak = 0
            self.quiz_fb.setText("✗ <b>Not quite.</b>  " + self.quiz["explain"])
            self.quiz_fb.setStyleSheet("color:#ef6461;")
            self.quiz_btns[i].setStyleSheet(bad_red)
            self.quiz_btns[c].setStyleSheet(ok_green)
        s = QSettings("KinLab2D", "progress")
        s.setValue("quizc", self.quiz_correct)
        s.setValue("quizt", self.quiz_total)
        s.setValue("quizbest", self.quiz_best)
        s.sync()
        self._quiz_score_update()

    def _quiz_show_pose(self):
        q = self.quiz
        if not (q and q.get("show")):
            return
        L, a = q["show"]
        if self.mode != "fk":
            self.rb_fk.setChecked(True)
        self.links_spin.blockSignals(True)
        self.links_spin.setValue(2)
        self.links_spin.blockSignals(False)
        self.lengths = [float(x) for x in L]
        self.angles = [x * D2R for x in a]
        self.pts = fk_positions(self.angles, self.lengths)
        self.rebuild_fk_ui()
        self.rebuild_len_ui()
        self.canvas.invalidate_ws()
        self.trail.clear()
        self.chart.clear()
        self.angle_plot.clear()
        self.refresh()

    def _goto_quiz(self):
        self.tabs.setCurrentWidget(self.quiz_page)
        self._quiz_new()

    # ------------------------------------------------------------- file I/O
    def save_scene(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save scene", "",
                                              "Scene (*.json)")
        if not path:
            return
        data = {
            "version": 4,
            "lengths": self.lengths,
            "angles_deg": [a * R2D for a in self.angles],
            "target": [float(self.target[0]), float(self.target[1])],
            "obstacles": [list(o) for o in self.obstacles],
            "pins": [[list(a), list(l)] for a, l in self.pins],
            "solver": self.solver_combo.currentIndex(),
            "tol": self.tol_spin.value(),
            "lam": self.lam_spin.value(),
            "limits": self.limits_spin.value() if self.limits_cb.isChecked() else 0,
            "mode": self.mode,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        self.statusBar().showMessage(f"saved {path}", 3000)

    def load_scene(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load scene", "",
                                              "Scene (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                d = json.load(f)
            n = len(d["lengths"])
            self.apply_setup(n, d["lengths"], d["angles_deg"],
                             mode=d.get("mode", "ik"),
                             solver=d.get("solver"), target=d["target"],
                             obstacles=d.get("obstacles", []),
                             tol=d.get("tol"), lam=d.get("lam"),
                             limits=d.get("limits", 0))
            self.pins = [(list(a), list(l))
                         for a, l in d.get("pins", [])]
            self.canvas.update()
            self.statusBar().showMessage(f"loaded {path}", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"load failed: {e}", 4000)

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "",
                                              "PNG (*.png)")
        if path:
            self.canvas.grab().save(path)
            self.statusBar().showMessage(f"saved {path}", 3000)

    def export_error_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export error log", "",
                                              "CSV (*.csv)")
        if not path:
            return
        with open(path, "w") as f:
            f.write("sample,error_u\n")
            for i, e in enumerate(self.chart.hist):
                f.write(f"{i},{e}\n")
        self.statusBar().showMessage(f"saved {path}", 3000)


# ----------------------------------------------------------------------------
# Dark theme
# ----------------------------------------------------------------------------
def apply_theme(app):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#1b1e26"))
    pal.setColor(QPalette.WindowText, QColor("#dfe3ee"))
    pal.setColor(QPalette.Base, QColor("#14161c"))
    pal.setColor(QPalette.AlternateBase, QColor("#1e222c"))
    pal.setColor(QPalette.Text, QColor("#dfe3ee"))
    pal.setColor(QPalette.Button, QColor("#262b37"))
    pal.setColor(QPalette.ButtonText, QColor("#dfe3ee"))
    pal.setColor(QPalette.Highlight, QColor("#3d6ddc"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#262b37"))
    pal.setColor(QPalette.ToolTipText, QColor("#dfe3ee"))
    pal.setColor(QPalette.PlaceholderText, QColor("#7b8296"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#7b8296"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#7b8296"))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#7b8296"))
    app.setPalette(pal)
    app.setStyleSheet("""
        QGroupBox { border: 1px solid #2b3140; border-radius: 8px;
                    margin-top: 14px; background: #1e222c; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px;
                           padding: 0 4px; color: #8fb7ff; }
        QTabWidget::pane { border: 1px solid #2b3140; border-radius: 6px; }
        QPushButton { background: #2a3140; border: 1px solid #3a4356;
                      border-radius: 6px; padding: 5px 10px; }
        QPushButton:hover { background: #343d52; }
        QPushButton:pressed { background: #3d6ddc; }
        QSlider::groove:horizontal { height: 4px; background: #333a4a;
                                     border-radius: 2px; }
        QSlider::handle:horizontal { width: 14px; margin: -5px 0;
                                     border-radius: 7px; background: #5b8cff; }
        QSlider::handle:horizontal:hover { background: #7ca3ff; }
        QRadioButton, QCheckBox { spacing: 6px; }
        QListWidget::item:selected { background: #3d6ddc; }
        QTableWidget { background: #14161c; gridline-color: #2b3140; }
        QHeaderView::section { background: #262b37; border: none;
                               padding: 4px; }
    """)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()