import numpy as np
import matplotlib.pyplot as plt
import scipy

def plot_trajs(position, velocity, jerk):
    dt = 0.02
    pos = position
    vel = np.diff(pos, n=1, axis=0) / dt
    acc = np.diff(vel, n=1, axis=0) / dt
    jer = np.diff(acc, n=1, axis=0) / dt

    pos = position[:-9]  # 141 steps for 2801 interp steps
    vel = velocity[:-9]
    acc = acc[:-7]
    jer = jer[:-6]
    jerk = jerk[:-9]

    # interp pos
    tf = dt
    prev_pos = position[0]
    prev_vel = velocity[0]
    prev_acc = np.array([0, 0, 0])
    interp_pos = [prev_pos]
    interp_vel = [prev_vel]
    interp_acc = [prev_acc]
    for i in range(position.shape[0] - 10):
        coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
        reg = np.array([[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 1e-4, 0], [0, 0, 0, 0.001]])
        # coef = coef + reg
        results = np.vstack([prev_pos, position[i+1], prev_vel, velocity[i+1]])
        A = scipy.linalg.block_diag(*[coef] * 3)
        y = results.reshape(-1, order='F')
        weights = np.linalg.solve(A, y).reshape(3, 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        prev_pos = np.polynomial.polynomial.polyval(tf, weights.T)
        prev_vel = np.polynomial.polynomial.polyval(tf, weights_d.T)
        prev_acc = np.polynomial.polynomial.polyval(tf, weights_dd.T)

        for t in np.linspace(0.001, 0.02, 20):
            q = np.polynomial.polynomial.polyval(t, weights.T)
            qd = np.polynomial.polynomial.polyval(t, weights_d.T)
            qdd = np.polynomial.polynomial.polyval(t, weights_dd.T)
            interp_pos.append(q)
            interp_vel.append(qd)
            interp_acc.append(qdd)

    step = np.linspace(dt, dt * pos.shape[0], pos.shape[0])
    constr_j_pos = [[-2.9, +2.9], [-1.8, +1.8], [-2.0, +2.0]]
    constr_j_vel = [[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]]
    constr_j_jerk = [[0, 1e4]] * 3

    interp_pos = np.array(interp_pos)
    interp_vel = np.array(interp_vel)
    interp_acc = np.array(interp_acc)
    interp_step = np.linspace(dt, dt * pos.shape[0], interp_pos.shape[0])

    s = 1
    ss = 20 * s
    e = 130
    ee = 20 * (e - 1) + 1
    for k in range(3):
        plt.subplot(3, 5, 5 * k + 1)
        if k == 0:
            plt.title("mp_pos vs. interp_pos")
        plt.plot(step[s:e], pos[s:e, k])
        plt.plot(interp_step[ss:ee], interp_pos[ss:ee, k])
        plt.hlines(constr_j_pos[k], xmin=step[0], xmax=step[-1], colors="r")

        plt.subplot(3, 5, 5 * k + 2)
        if k == 0:
            plt.title("mp_vel vs. interp_vel")
        plt.plot(step[s:e], vel[s:e, k])
        plt.plot(interp_step[ss:ee], interp_vel[ss:ee, k])
        plt.hlines(constr_j_vel[k], xmin=step[0], xmax=step[-1], colors="r")

        plt.subplot(3, 5, 5 * k + 3)
        if k == 0:
            plt.title("mp_acc vs. interp_acc")
        plt.plot(interp_step[ss:ee], interp_acc[ss:ee, k])
        plt.plot(step[s:e], acc[s:e, k])

        plt.subplot(3, 5, 5 * k + 4)
        if k == 0:
            plt.title("mp_jerk")
        plt.plot(step[2:e], jer[2:e, k])

        plt.subplot(3, 5, 5 * k + 5)
        if k == 0:
            plt.title("interp_jerk")
        plt.plot(step[2:e], jerk[2:e, k])
        plt.hlines(constr_j_jerk[k], xmin=step[0], xmax=step[-1], colors='r')
    plt.show()

def plot_real_vs_supposed(positions, supposed_positions, velocities, supposed_velocities):
    positions = np.array(positions)
    supposed_positions = np.array(supposed_positions)
    velocities = np.array(velocities)
    supposed_velocities = np.array(supposed_velocities)
    for k in range(3):
        plt.subplot(3, 2, 2 * k + 1)
        if k == 0:
            plt.title("env_pos vs. supposed_pos")
        plt.plot(list(range(positions.shape[0])), positions[:, k])
        plt.plot(list(range(supposed_positions.shape[0])), supposed_positions[:, k])

        plt.subplot(3, 2, 2 * k + 2)
        if k == 0:
            plt.title("env_vel vs. supposed_vel")
        plt.plot(list(range(velocities.shape[0])), velocities[:, k])
        plt.plot(list(range(supposed_velocities.shape[0])), supposed_velocities[:, k])

    plt.show()


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
import fancy_gym

# from baseline.baseline_agent.baseline_agent import build_agent
# if __name__ == "__main__":
#     env = fancy_gym.make("3dof-hit-interpolate", seed=2)
#     env_info = env.env_info
#     agent = build_agent(env_info)
#     while True:
#         agent.reset()
#         obs = env.reset()
#         org_obs = obs
#         positions = []
#         velocities = []
#         jerks = []
#         supposed_pos = []
#         supposed_vel = []
#         for i in range(1000):
#             action = agent.draw_action(obs).reshape([-1])
#             obs, rew, done, info = env.step(action)
#             positions.append(obs[[6,7,8]])
#             velocities.append(obs[[9,10,11]])
#             jerks.append(info["jerk"])
#             supposed_pos.append(info["supposed_pos"])
#             supposed_vel.append(info["supposed_vel"])
#             env.render()
#             if done:
#                 plot_real_vs_supposed(positions, supposed_pos, velocities, supposed_vel)
#                 # plot_trajs(np.array(positions), np.array(velocities), np.array(jerks))
#                 exit()

if __name__ == "__main__":
    name = "16k"
    model_load = f"logs/{name}/best_model.zip"
    vecnormalize_load = f"logs/{name}/vec_normalize_data"
    eval_env = VecMonitor(DummyVecEnv([lambda: fancy_gym.make("3dof-hit-interpolate", seed=3)]))
    eval_env = VecNormalize.load(vecnormalize_load, eval_env)
    model = PPO.load(model_load)
    while True:
        obs = eval_env.reset()
        org_obs = obs
        positions = []
        velocities = []
        jerks = []
        supposed_pos = []
        supposed_vel = []
        for i in range(1000):
            action = model.predict(obs, deterministic=True)
            action = [np.reshape([20,20,20,0,0,0], [2,-1])]
            obs, rew, done, info = eval_env.step(action)
            print(obs[0][6:])
            positions.append(obs[0][[6,7,8]])
            velocities.append(obs[0][[9,10,11]])
            jerks.append(info[0]["jerk"])
            supposed_pos.append(info[0]["supposed_pos"])
            supposed_vel.append(info[0]["supposed_vel"])
            eval_env.render()
            if done[0]:
                a = input()
                if a == "y":
                    print(i)
                    plot_real_vs_supposed(positions, supposed_pos, velocities, supposed_vel)
                    # plot_trajs(np.array(positions), np.array(velocities), np.array(jerks))
                break