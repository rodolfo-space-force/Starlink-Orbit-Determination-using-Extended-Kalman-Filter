#Rodolfo Milhomem
#https://github.com/rodolfo-space-force/

import numpy as np
import matplotlib.pyplot as plt
import requests
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
import re

# ==========================================================
# CONSTANTES FÍSICAS (km, s)
# ==========================================================

MU = 398600.4418
J2 = 1.08263e-3
RE = 6378.137

# ==========================================================
# CONFIGURAÇÃO: ESCOLHER NORAD ID
# ==========================================================

TARGET_NORAD_ID = "44714"   # <-- coloque aqui o NORAD desejado

# ==========================================================
# 1. DOWNLOAD TLE STARLINK POR NORAD ID
# ==========================================================

def get_starlink_tle_by_norad(norad_id):

    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    response = requests.get(url)
    response.raise_for_status()
    data = response.text.splitlines()

    for i in range(len(data)-2):
        if data[i+1].startswith("1 ") and data[i+2].startswith("2 "):

            line1 = data[i+1]
            line2 = data[i+2]

            # NORAD ID está nas colunas 3–7 da linha 1
            extracted_id = line1[2:7].strip()

            if extracted_id == norad_id:
                print("Using satellite:", data[i])
                print("NORAD ID:", extracted_id)
                return line1, line2

    raise RuntimeError(f"NORAD ID {norad_id} not found in Starlink TLE list.")

# ==========================================================
# 2. SGP4 TRUTH MODEL
# ==========================================================

def propagate_sgp4(line1, line2, duration=6*3600, step=30):

    sat = Satrec.twoline2rv(line1, line2)
    t0 = datetime.now(timezone.utc)

    states = []

    for t in range(0, duration, step):
        current_time = t0 + timedelta(seconds=t)

        jd, fr = jday(current_time.year, current_time.month,
                      current_time.day, current_time.hour,
                      current_time.minute, current_time.second)

        error, r, v = sat.sgp4(jd, fr)

        if error == 0:
            states.append(np.hstack((r, v)))

    if len(states) == 0:
        raise RuntimeError("SGP4 propagation failed.")

    return np.array(states)

# ==========================================================
# 3. DINÂMICA KEPLER + J2
# ==========================================================

def acceleration_j2(r):
    x, y, z = r
    r_norm = np.linalg.norm(r)

    factor = 1.5 * J2 * MU * RE**2 / r_norm**5
    zx = 5 * z**2 / r_norm**2

    ax = factor * x * (zx - 1)
    ay = factor * y * (zx - 1)
    az = factor * z * (zx - 3)

    return np.array([ax, ay, az])

def dynamics(state):
    r = state[:3]
    v = state[3:]

    r_norm = np.linalg.norm(r)

    a_kepler = -MU * r / r_norm**3
    a = a_kepler + acceleration_j2(r)

    return np.hstack((v, a))

# ==========================================================
# 4. RK4 INTEGRATOR
# ==========================================================

def rk4_step(x, dt):
    k1 = dynamics(x)
    k2 = dynamics(x + 0.5 * dt * k1)
    k3 = dynamics(x + 0.5 * dt * k2)
    k4 = dynamics(x + dt * k3)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# ==========================================================
# 5. NUMERICAL JACOBIAN
# ==========================================================

def compute_F_numeric(x, dt):
    eps = 1e-6
    F = np.zeros((6, 6))

    for i in range(6):
        dx = np.zeros(6)
        dx[i] = eps
        F[:, i] = (rk4_step(x + dx, dt) -
                   rk4_step(x - dx, dt)) / (2 * eps)

    return F

# ==========================================================
# 6. EKF
# ==========================================================

H = np.hstack((np.eye(3), np.zeros((3,3))))

def ekf_predict(x, P, Q, dt):
    x_pred = rk4_step(x, dt)
    F = compute_F_numeric(x, dt)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x, P, z, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_upd = x + K @ y
    P_upd = (np.eye(6) - K @ H) @ P
    return x_upd, P_upd

# ==========================================================
# 7. SINGLE MONTE CARLO RUN
# ==========================================================

def single_run(truth, dt=30):

    n = len(truth)

    x_est = truth[0] + np.hstack((
        np.random.normal(0, 0.1, 3),     # 100 m
        np.random.normal(0, 0.0001, 3)   # 0.1 m/s
    ))

    P = np.diag([1e-2]*3 + [1e-6]*3)
    Q = np.diag([1e-9]*3 + [1e-12]*3)
    R = np.diag([0.05**2]*3)

    errors = []

    for k in range(n):

        z = truth[k, :3] + np.random.normal(0, 0.05, 3)

        x_est, P = ekf_predict(x_est, P, Q, dt)
        x_est, P = ekf_update(x_est, P, z, R)

        err = np.linalg.norm(x_est[:3] - truth[k, :3])
        errors.append(err)

    return np.array(errors)

# ==========================================================
# 8. MAIN MONTE CARLO
# ==========================================================

if __name__ == "__main__":

    line1, line2 = get_starlink_tle_by_norad(TARGET_NORAD_ID)
    truth = propagate_sgp4(line1, line2)

    runs = 20
    dt = 30
    all_errors = []

    for i in range(runs):
        print(f"Monte Carlo run {i+1}/{runs}")
        errors = single_run(truth, dt)
        all_errors.append(errors)

    all_errors = np.array(all_errors) * 1000  # km → m

    mean_error = np.mean(all_errors, axis=0)
    std_error = np.std(all_errors, axis=0)

    rms_global = np.sqrt(np.mean(all_errors**2))

    discard_steps = 200
    steady_state = all_errors[:, discard_steps:]
    rms_steady = np.sqrt(np.mean(steady_state**2))

    convergence_index = np.argmax(mean_error < 500)
    convergence_time_sec = convergence_index * dt

    print("\n===== PERFORMANCE SUMMARY =====")
    print("NORAD ID:", TARGET_NORAD_ID)
    print("Global RMS (m):", rms_global)
    print("Steady-State RMS (m):", rms_steady)
    print("Approx. Convergence Time (s):", convergence_time_sec)
    print("================================\n")

    plt.figure(figsize=(10,5))
    plt.plot(mean_error, label="Mean Error")
    plt.fill_between(
        range(len(mean_error)),
        mean_error - std_error,
        mean_error + std_error,
        alpha=0.3,
        label="±1σ"
    )
    plt.axvline(discard_steps, linestyle="--", color="red", label="Steady-State Start")
    plt.ylabel("Position Error (m)")
    plt.xlabel("Time Step")
    plt.title(f"Monte Carlo EKF Performance - NORAD {TARGET_NORAD_ID}")
    plt.legend()
    plt.grid()
    plt.show()

# Licença
#Este projeto está licenciado sob a **Licença MIT**.  
#Você pode usar, modificar e redistribuir este código livremente, **desde que mencione o autor original**.
