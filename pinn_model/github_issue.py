from deepxde.icbc import DirichletBC
from deepxde.geometry.geometry_2d import Polygon
from deepxde.callbacks import EarlyStopping
from deepxde.nn import FNN
from deepxde.data.pde import PDE
from deepxde.model import Model
from deepxde.backend import tf
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import torch


def wall_top_boundary(x, on_boundary):
    """Checks for points on top wall boundary"""
    return on_boundary and np.isclose(x[1], 2.0)


def wall_bottom_boundary(x, on_boundary):
    """Checks for points on bottom wall boundary"""
    return on_boundary and np.isclose(x[1], 0.0)


def wall_mid_horizontal_boundary(x, on_boundary):
    """Check for points on step horizontal boundary"""
    return on_boundary and (np.isclose(x[1], 1.0) and x[0] < 2.0)


def wall_mid_vertical_boundary(x, on_boundary):
    """Check for points on step horizontal boundary"""
    return on_boundary and (x[1] < 1.0 and np.isclose(x[0], 2.0))


def outlet_boundary(x, on_boundary):
    """Implements the outlet boundary with zero y-velocity component"""
    return on_boundary and np.isclose(x[0], 12.0)


def inlet_boundary(x, on_boundary):
    """Implements the inlet boundary with parabolic x-velocity component"""
    return on_boundary and np.isclose(x[0], 0.0)

def boundary_circle(x, on_boundary):
    #centre (4, 1), radius 0.3
    return on_boundary and disk_domain.on_boundary(x)


def parabolic_velocity(x):
    """Parabolic velocity"""
    return (6 * (x[:, 1] - 1) * (2 - x[:, 1])).reshape(-1, 1)


def zero_velocity(x):
    """Zero velocity"""
    return np.zeros((x.shape[0], 1))




def navier_stokes(x, u):
    #x - x, y coordinates
    #u - u, v, p
    rho = 1
    mu = 0.01
    eps = 1e-8
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        rho * (u_vel * u_vel_x + v_vel * u_vel_y) + p_x - mu * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        rho * (u_vel * v_vel_x + v_vel * v_vel_y) + p_y - mu * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]

def output_transformer(inputs, outputs):
    """Apply output transforms to strictly enforce boundary conditions"""

    top_line = inputs[:, 1] - 2.0
    bottom_line = inputs[:, 1]
    mid_hor_line = inputs[:, 1] - 1.0
    mid_ver_line = inputs[:, 0] - 2.0
    outlet_line = inputs[:, 0] - 12.0
    inlet_line = inputs[:, 0]
    velocity_profile = 6.0 * (inputs[:, 1] - 1.0) * (2.0 - inputs[:, 1])
    # velocity_profile = 1.0

    u_multiplier = (top_line * bottom_line * mid_hor_line * mid_ver_line *
                    velocity_profile)
    v_multiplier = (top_line * bottom_line * mid_hor_line * mid_ver_line *
                    outlet_line * inlet_line)
    p_multiplier = 1.0

    return torch.cat([(u_multiplier * outputs[:, 0]).reshape(-1, 1), (v_multiplier * outputs[:, 1]).reshape(-1, 1), (p_multiplier * outputs[:, 2]).reshape(-1, 1)], dim=1)



if __name__ == '__main__':
    """
    Geometry
    --------
                (0, 2)       (12, 2)
                  *------------*
        in -> |            |
        (0, 1)*--*(2,1)    . -> out
                    |         |
            (2,0)*---------*(12, 0)
    """
    geom = Polygon([
        [0.0, 2.0], [12.0, 2.0], [12.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [0.0, 1.0]
    ])
    disk_domain = dde.geometry.Disk([4, 1], 0.3)
    geom = geom - disk_domain

    inlet_x = DirichletBC(geom, parabolic_velocity, inlet_boundary,
                          component=0)
    inlet_y = DirichletBC(geom, zero_velocity, inlet_boundary, component=1)
    outlet = DirichletBC(geom, zero_velocity, outlet_boundary, component=1)
    wallt_x = DirichletBC(geom, zero_velocity, wall_top_boundary, component=0)
    wallt_y = DirichletBC(geom, zero_velocity, wall_top_boundary, component=1)
    wallb_x = DirichletBC(geom, zero_velocity, wall_bottom_boundary,
                          component=0)
    wallb_y = DirichletBC(geom, zero_velocity, wall_bottom_boundary,
                          component=1)
    wallsh_x = DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary,
                           component=0)
    wallsh_y = DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary,
                           component=1)
    wallsv_x = DirichletBC(geom, zero_velocity, wall_mid_vertical_boundary,
                           component=0)
    wallsv_y = DirichletBC(geom, zero_velocity, wall_mid_vertical_boundary,
                           component=1)
    circle = DirichletBC(geom, zero_velocity, boundary_circle)


    data = PDE(
        geom, navier_stokes,
        [inlet_x, inlet_y, outlet, wallt_x, wallt_y, wallb_x,
        wallb_x, wallsh_x, wallsh_y, wallsv_x, wallsv_y, circle], num_domain=50000,
        num_boundary=50000, num_test=10000
    )

    layer_size = [2] + [50] * 6 + [3]
    net = FNN(layer_size, "tanh", "Glorot uniform")
    # net.apply_output_transform(output_transformer)

    model = Model(data, net)
    model.compile("adam", lr=0.001)

    # early_stopping = EarlyStopping(min_delta=1e-8, patience=40000)
    # checker = dde.callbacks.ModelCheckpoint("model/bend_model.ckpt", save_better_only=True, period=1000)
    # model.train(iterations=5000, display_every=1000, callbacks=[checker])

    model.compile("L-BFGS")
    # losshistory, train_state = model.train(model_save_path = "./model/bend_model.ckpt")

    # model.restore("model/bend_model.ckpt-20000.pt", verbose=1)  # Replace ? with the exact filename
    X = geom.random_points(100000)
    output = model.predict(X)

    u_pred = output[:, 0]
    v_pred = output[:, 1]
    velocity_pred = np.sqrt(u_pred**2 + v_pred**2)
    p_pred = output[:, 2]

    fig = plt.figure(figsize=(12,2))
    plt.scatter(X[:, 0], X[:, 1], c=velocity_pred, cmap='jet',vmin=0, vmax=1, marker = '.')
    plt.colorbar().ax.set_ylabel('z', rotation=270)


    #residual plot 
    f = model.predict(X, operator=navier_stokes)
    print("f shape:", f[0].shape)
    print("X shape:", X.shape)

    fig = plt.figure(figsize=(12,2))
    plt.scatter(X[:, 0], X[:, 1], c=f[0], cmap='jet',vmin=0, vmax=1, marker = '.')
    plt.colorbar().ax.set_ylabel('z', rotation=270)
    plt.show()


    residual = np.mean(np.absolute(f))

    print("Mean residual:", residual)