"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np


dde.backend.set_default_backend("tensorflow.compat.v1")

import matplotlib.pyplot as plt

a = 1
d = 1
Re = 1


def pde(x, u):
    u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
    u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
    v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

    w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
    w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
    w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
    w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
    w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
    w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
    w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

    p_x = dde.grad.jacobian(u, x, i=3, j=0)
    p_y = dde.grad.jacobian(u, x, i=3, j=1)
    p_z = dde.grad.jacobian(u, x, i=3, j=2)

    momentum_x = (
        u_vel_t
        + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
        + p_x
        - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
    )
    momentum_y = (
        v_vel_t
        + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
        + p_y
        - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
    )
    momentum_z = (
        w_vel_t
        + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
        + p_z
        - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
    )
    continuity = u_vel_x + v_vel_y + w_vel_z

    return [momentum_x, momentum_y, momentum_z, continuity]


def u_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def v_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def w_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def p_func(x):
    return (
        -0.5
        * a ** 2
        * (
            np.exp(2 * a * x[:, 0:1])
            + np.exp(2 * a * x[:, 1:2])
            + np.exp(2 * a * x[:, 2:3])
            + 2
            * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
            + 2
            * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
            + 2
            * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
        )
        * np.exp(-2 * d ** 2 * x[:, 3:4])
    )


spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

boundary_condition_u = dde.DirichletBC(
    spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.DirichletBC(
    spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_w = dde.DirichletBC(
    spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
)

initial_condition_u = dde.IC(
    spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
)
initial_condition_v = dde.IC(
    spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
)
initial_condition_w = dde.IC(
    spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
)

data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [
        boundary_condition_u,
        boundary_condition_v,
        boundary_condition_w,
        initial_condition_u,
        initial_condition_v,
        initial_condition_w,
    ],
    num_domain=50000,
    num_boundary=5000,
    num_initial=5000,
    num_test=10000,
)

net = dde.nn.FNN([4] + 4 * [50] + [4], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
# model.train(iterations=30000, display_every=1)
model.compile("L-BFGS", loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
# losshistory, train_state = model.train()
model.restore("model/good_model.ckpt-43904.ckpt", verbose=1)
# losshistory, train_state = model.train(iterations=1, model_restore_path="model/good_model.ckpt-43904.ckpt")


# x, y, z = np.meshgrid(
#     np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
# )

# X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

# t_0 = np.zeros(1000).reshape(1000, 1)
# t_1 = np.ones(1000).reshape(1000, 1)

# X_0 = np.hstack((X, t_0))

# output_0 = model.predict(X_0)

# u_pred_0 = output_0[:, 0].reshape(-1)
# v_pred_0 = output_0[:, 1].reshape(-1)
# w_pred_0 = output_0[:, 2].reshape(-1)
# p_pred_0 = output_0[:, 3].reshape(-1)

# u_exact_0 = u_func(X_0).reshape(-1)
# v_exact_0 = v_func(X_0).reshape(-1)
# w_exact_0 = w_func(X_0).reshape(-1)
# p_exact_0 = p_func(X_0).reshape(-1)

# f_0 = model.predict(X_0, operator=pde)

# l2_difference_u_0 = dde.metrics.l2_relative_error(u_exact_0, u_pred_0)
# l2_difference_v_0 = dde.metrics.l2_relative_error(v_exact_0, v_pred_0)
# l2_difference_w_0 = dde.metrics.l2_relative_error(w_exact_0, w_pred_0)
# l2_difference_p_0 = dde.metrics.l2_relative_error(p_exact_0, p_pred_0)
# residual_0 = np.mean(np.absolute(f_0))

# print("Accuracy at t = 0:")
# print("Mean residual:", residual_0)
# print("L2 relative error in u:", l2_difference_u_0)
# print("L2 relative error in v:", l2_difference_v_0)
# print("L2 relative error in w:", l2_difference_w_0)

# # Визуализация решения
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Solution of Navier-Stokes Equations')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

# X, Y, Z = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))

# u_pred_0 = output_0[:, 0].reshape(X.shape)
# v_pred_0 = output_0[:, 1].reshape(X.shape)
# w_pred_0 = output_0[:, 2].reshape(X.shape)

# # Визуализация решения для t = 0
# ax.quiver(X, Y, Z, u_pred_0, v_pred_0, w_pred_0, length=0.1, normalize=True, color='r')
# ax.set_title('Solution at t = 0')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Assuming you have defined model, u_func, v_func, w_func, p_func, and pde 

# fig = plt.figure()
# ax : plt.Axes = fig.add_subplot(111, projection='3d')

# # Set up meshgrid
# x, y, z = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
# X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

# # Time parameters
# t_values = np.linspace(0, 1, 60)  # 60 frames for the animation

# # Initialize quiver plot
# quiver = ax.quiver(x, y, z, np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape), length=0.1, normalize=True, color='r')

# # Set plot labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Solution of Navier-Stokes Equations')

# # Function to update the plot for each frame
# def update(frame):
#     t = t_values[frame]
#     t_array = np.full((1000, 1), t)
#     X_t = np.hstack((X, t_array))

#     output = model.predict(X_t)
#     u_pred = output[:, 0].reshape(x.shape)
#     v_pred = output[:, 1].reshape(y.shape)
#     w_pred = output[:, 2].reshape(z.shape)

#     # Update quiver data
#     new_segs = [[[x_, y_, z_], [u, v, w]] for x_, y_, z_, u, v, w in 
#                  zip(x, y, z, u_pred, v_pred, w_pred)]
#     print(np.array(new_segs).shape)
#     quiver.set_segments(new_segs)

#     # Update title
#     ax.set_title(f'Solution at t = {t:.2f}')
#     print(t)
#     return quiver,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(t_values), blit=False, interval=50)
# ani.save("cube_model.gif", writer="imagemagick", fps=10)


# Assuming x, y, z are 1D arrays representing coordinates
# If they are meshgrids, flatten them:
# x = x.flatten()
# y = y.flatten()
# z = z.flatten()

# # Initialize quiver plot
# quiver = ax.quiver(x, y, z, np.zeros(x.size), np.zeros(y.size), np.zeros(z.size), length=0.1, normalize=True, color='r')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Solution of Navier-Stokes Equations')

# def update(frame):
#     t = t_values[frame]
#     t_array = np.full((1000, 1), t)
#     X_t = np.hstack((X, t_array))

#     output = model.predict(X_t)
#     u_pred = output[:, 0].reshape(x.size)  # Use x.size
#     v_pred = output[:, 1].reshape(y.size)  # Use y.size
#     w_pred = output[:, 2].reshape(z.size)  # Use z.size

#     # Update quiver data (optimized)
#     scale = 0.1
#     new_segs = [[[xi, yi, zi], [xi + scale * ui, yi + scale * vi, zi + scale * wi]] for xi, yi, zi, ui, vi, wi in zip(x, y, z, u_pred, v_pred, w_pred)]

#     quiver.set_segments(new_segs)
#     ax.set_title(f'Solution at t = {t:.2f}')

#     return quiver,

# ani = FuncAnimation(fig, update, frames=len(t_values), blit=False, interval=50)
# ani.save("cube_model.gif", writer="imagemagick", fps=10)

# plt.show()




# # Create the figure and 2D axes
# fig = plt.figure()
# ax = fig.add_subplot(111)


# x, y, z = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
# X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

# # Find indices where Z is closest to 0.5
# z_target = 0.5
# z_indices = np.argmin(np.abs(z - z_target))

# x2, y2 = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))

# # Initialize quiver plot
# quiver = ax.quiver(x2, y2, np.zeros(x2.shape), np.zeros(y2.shape), color='r')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Solution of Navier-Stokes Equations at Z = 0.5')

# def update(frame):
#     t = t_values[frame]
#     t_array = np.full((1000, 1), t)
#     X_t = np.hstack((X, t_array))

#     output = model.predict(X_t)
#     u_pred = output[:, 0].reshape(x.shape)
#     v_pred = output[:, 1].reshape(y.shape)

#     # Extract u and v at Z = 0.5
#     u_2d = u_pred[z_indices, :, :]
#     v_2d = v_pred[z_indices, :, :]

#     # Update quiver data
#     quiver.set_UVC(u_2d, v_2d)
#     ax.set_title(f'Solution at t = {t:.2f} (Z = 0.5)')

#     return quiver,

# ani = FuncAnimation(fig, update, frames=len(t_values), blit=False, interval=50)
# ani.save("2d_quiver_z05.gif", writer="imagemagick", fps=10)

# plt.show()

# dde.utils.plot_loss_history(losshistory)
# plt.show()


# import time

# steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# result = []
# for step in steps:
#     x, y, z = np.meshgrid(np.linspace(-1, 1, step), np.linspace(-1, 1, step), np.linspace(-1, 1, step))
#     X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

#     # Time parameters
#     t_values = np.linspace(0, 1, 60)  # 60 frames for the animation


#     start = time.time()
#     shape = X.shape[0]

#     for t in t_values:

#         t_array = np.full((shape, 1), t)
#         X_t = np.hstack((X, t_array))

#         output = model.predict(X_t)

#     result.append(time.time() - start)
#     print(result[-1])

# plt.plot(np.array([result, steps]).reshape(-1, 2))
# plt.show()

import time
import statistics
import matplotlib.pyplot as plt

def calculate_model(X, t):
    t_array = np.full((X.shape[0], 1), t)
    X_t = np.hstack((X, t_array))
    model.predict(X_t)

t = 60
n_try = 10

params = [  
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
]

runtimes = []
for k in range(len(params)):
    x, y, z = np.meshgrid(np.linspace(-1, 1, params[k]), np.linspace(-1, 1, params[k]), np.linspace(-1, 1, params[k]))
    X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
    runtimes.append([])
    for i in range(n_try):
        start_time = time.time()
        calculate_model(X, t)
        end_time = time.time()
        runtimes[k].append(end_time - start_time)
    print(params[k])

mean_runtimes = [statistics.mean(runtime) for runtime in runtimes]
std_dev = [statistics.stdev(runtime) for runtime in runtimes]

plt.figure()
plt.errorbar(np.array(params) ** 3, mean_runtimes, yerr=std_dev, fmt='o', capsize=2)
plt.xlabel("Количество точек")
plt.ylabel("Время выполнения (сек)")
plt.title("Время выполнения модели в зависимости от количества точек")
plt.show()
