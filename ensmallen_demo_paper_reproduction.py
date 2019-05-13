# 2019-05-13
# purpose of this script is to reproduce the experiments show in the article:
# https://new.qq.com/omn/20190328/20190328A00CIC.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# returns the function´s value for given x,y
def booth(x, y):
    return np.square(x + 2*y - 7)+np.square(2*x + y - 5)


# returns the function´s gradient for given x,y
def booth_grad(x, y):
    x_grad = 8*y + 10*x - 34
    y_grad = 8*x + 10*y - 38
    return x_grad, y_grad


# returns the function´s value for given x,y
def styblinski_tang(x, y):
    return (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y) / 2


# returns the function´s gradient for given x,y
def styblinski_tang_grad(x, y):
    x_grad = (4*x**3 - 32*x + 5) / 2
    y_grad = (4*y**3 - 32*y + 5) / 2
    return x_grad, y_grad


# runs the Adam optimizer on a given loss function with given options.
# returns x,y and loss function for every iteration step.
def adam(x_initial, y_initial, total_iterations, step_size, loss_function, beta_1, beta_2, eps):
    # creating vectors used in loop and setting initial values for adam
    x_all = np.zeros(total_iterations + 1)
    y_all = np.zeros(total_iterations + 1)
    x_all[0] = x_initial
    y_all[0] = y_initial
    m_x_all = np.zeros(total_iterations + 1)  # value for t = 0 should be 0 for m and v
    m_y_all = np.zeros(total_iterations + 1)
    v_x_all = np.zeros(total_iterations + 1)
    v_y_all = np.zeros(total_iterations + 1)
    loss_all = np.zeros(total_iterations + 1)
    loss_grad_x_all = np.zeros(total_iterations + 1)
    loss_grad_y_all = np.zeros(total_iterations + 1)
    if loss_function == "booth":
        loss_all[0] = booth(x_all[0], y_all[0])
        loss_grad_x_all[0], loss_grad_y_all[0] = booth_grad(x_all[0], y_all[0])
    elif loss_function == "styblinski_tang":
        loss_all[0] = styblinski_tang(x_all[0], y_all[0])
        loss_grad_x_all[0], loss_grad_y_all[0] = styblinski_tang_grad(x_all[0], y_all[0])
    else:
        return False
    # start iterating
    for t in range(total_iterations):
        m_x_all[t + 1] = beta_1 * m_x_all[t] + (1 - beta_1) * loss_grad_x_all[t]
        m_y_all[t + 1] = beta_1 * m_y_all[t] + (1 - beta_1) * loss_grad_y_all[t]
        v_x_all[t + 1] = beta_2 * v_x_all[t] + (1 - beta_2) * np.square(loss_grad_x_all[t])
        v_y_all[t + 1] = beta_2 * v_y_all[t] + (1 - beta_2) * np.square(loss_grad_y_all[t])
        mh_x = m_x_all[t + 1] / (1 - beta_1 ** (t + 1))
        mh_y = m_y_all[t + 1] / (1 - beta_1 ** (t + 1))
        vh_x = v_x_all[t + 1] / (1 - beta_2 ** (t + 1))
        vh_y = v_y_all[t + 1] / (1 - beta_2 ** (t + 1))
        x_all[t + 1] = x_all[t] - step_size * mh_x / (np.sqrt(vh_x) + eps)
        y_all[t + 1] = y_all[t] - step_size * mh_y / (np.sqrt(vh_y) + eps)
        if loss_function == "booth":
            loss_all[t + 1] = booth(x_all[t + 1], y_all[t + 1])
            loss_grad_x_all[t + 1], loss_grad_y_all[t + 1] = booth_grad(x_all[t + 1], y_all[t + 1])
        elif loss_function == "styblinski_tang":
            loss_all[t + 1] = styblinski_tang(x_all[t + 1], y_all[t + 1])
            loss_grad_x_all[t + 1], loss_grad_y_all[t + 1] = styblinski_tang_grad(x_all[t + 1], y_all[t + 1])
    return{"x_all": x_all, "y_all": y_all, "loss_all": loss_all}


# runs the AdaGrad optimizer on a given loss function with given options.
# returns x,y and loss function for every iteration step.
def ada_grad(x_initial, y_initial, total_iterations, step_size, loss_function):
    # creating vectors used in loop and setting initial values for adam
    x_all = np.zeros(total_iterations + 1)
    y_all = np.zeros(total_iterations + 1)
    x_all[0] = x_initial
    y_all[0] = y_initial
    loss_all = np.zeros(total_iterations + 1)
    g_x_all = np.zeros(total_iterations + 1)  # more memory efficient to make aa constant instead of vector.
    g_y_all = np.zeros(total_iterations + 1)
    G_diag_x_all = np.zeros(total_iterations + 1)  # also only saved as a vector for testing purposes
    G_diag_y_all = np.zeros(total_iterations + 1)
    if loss_function == "booth":
        loss_all[0] = booth(x_all[0], y_all[0])
        g_x_all[0], g_y_all[0] = booth_grad(x_all[0], y_all[0])
    elif loss_function == "styblinski_tang":
        loss_all[0] = styblinski_tang(x_all[0], y_all[0])
        g_x_all[0], g_y_all[0] = styblinski_tang_grad(x_all[0], y_all[0])
    else:
        return False
    G_diag_x_all[0] = np.square(g_x_all[0])
    G_diag_y_all[0] = np.square(g_y_all[0])
    # start iterating
    for t in range(total_iterations):
        # update x,y
        x_all[t + 1] = x_all[t] - step_size * g_x_all[t] / np.sqrt(G_diag_x_all[t])
        y_all[t + 1] = y_all[t] - step_size * g_y_all[t] / np.sqrt(G_diag_y_all[t])
        # update g, G preparing for next iteration. Also calculates loss function
        if loss_function == "booth":
            loss_all[t + 1] = booth(x_all[t + 1], y_all[t + 1])
            g_x_all[t + 1], g_y_all[t + 1] = booth_grad(x_all[t + 1], y_all[t + 1])
        elif loss_function == "styblinski_tang":
            loss_all[t + 1] = styblinski_tang(x_all[t + 1], y_all[t + 1])
            g_x_all[t + 1], g_y_all[t + 1] = styblinski_tang_grad(x_all[t + 1], y_all[t + 1])
        G_diag_x_all[t + 1] = G_diag_x_all[t] + np.square(g_x_all[t + 1])
        G_diag_y_all[t + 1] = G_diag_y_all[t] + np.square(g_y_all[t + 1])
    return {"x_all": x_all, "y_all": y_all, "loss_all": loss_all}


def create_booth_matrix(x_mesh, y_mesh):
    return np.square(x_mesh + 2*y_mesh - 7) + np.square(2*x_mesh + y_mesh - 5)


def create_styblinski_tang_matrix(x_mesh, y_mesh):
    return ((x_mesh**4 - 16*x_mesh**2 + 5*x_mesh) + (y_mesh**4 - 16*y_mesh**2 + 5*y_mesh)) / 2


# variables for plotting
graph_points = 151
x_plot_vector_booth = np.linspace(-10, 10, graph_points)
y_plot_vector_booth = np.linspace(-10, 10, graph_points)
x_mesh_booth, y_mesh_booth = np.meshgrid(x_plot_vector_booth, y_plot_vector_booth)
x_plot_vector_styblinski_tang = np.linspace(-5, 5, graph_points)
y_plot_vector_styblinski_tang = np.linspace(-5, 5, graph_points)
x_mesh_styblinski_tang, y_mesh_styblinski_tang = np.meshgrid(x_plot_vector_styblinski_tang,
                                                             y_plot_vector_styblinski_tang)
booth_matrix = create_booth_matrix(x_mesh_booth, y_mesh_booth)
styblinski_matrix = create_styblinski_tang_matrix(x_mesh_styblinski_tang, y_mesh_styblinski_tang)


# plots booth function with colors, x and y denote optimization path
def plot_booth(x, y):
    plt.pcolor(x_mesh_booth, y_mesh_booth, create_booth_matrix(x_mesh_booth, y_mesh_booth))
    plt.plot(x, y)
    plt.contour(x_mesh_booth, y_mesh_booth, create_booth_matrix(x_mesh_booth, y_mesh_booth))
    plt.xlabel("x")
    plt.ylabel("y")


# plots tyblinski-tang function with colors, x and y denote optimization path
def plot_tyblinski_tang(x, y):
    plt.pcolor(x_mesh_styblinski_tang, y_mesh_styblinski_tang, create_styblinski_tang_matrix(x_mesh_styblinski_tang,
                                                                                             y_mesh_styblinski_tang))
    plt.plot(x, y)
    plt.contour(x_mesh_styblinski_tang, y_mesh_styblinski_tang, create_styblinski_tang_matrix(x_mesh_styblinski_tang,
                                                                                              y_mesh_styblinski_tang))
    plt.xlabel("x")
    plt.ylabel("y")


# plots loss function against iterations
def plot_loss(loss):
    x = np.linspace(0, np.shape(loss)[0], np.shape(loss)[0])
    plt.plot(x, loss)
    plt.xlabel("iterations")
    plt.ylabel("loss function")


# run the optimizations and plots.
def run_demo():
    # time step investigation
    return_dict1 = adam(x_initial=-4, y_initial=-4, total_iterations=1000, step_size=0.3, loss_function="booth",
                        beta_1=0.9, beta_2=0.999, eps=0.00001)
    x1 = return_dict1["x_all"]
    y1 = return_dict1["y_all"]
    print(1, x1[-1])
    print(1, y1[-1])
    plt.figure()
    plt.title("Adam optimiser on Booth function, step size = 0.3")
    plot_booth(x1, y1)
    return_dict2 = adam(x_initial=-4, y_initial=-4, total_iterations=1000, step_size=0.03, loss_function="booth",
                        beta_1=0.9, beta_2=0.999, eps=0.00001)
    x2 = return_dict2["x_all"]
    y2 = return_dict2["y_all"]
    print(2, x1[-1])
    print(2, y1[-1])
    plt.figure()
    plt.title("Adam optimiser on Booth function, step size = 0.03")
    plot_booth(x2, y2)
    return_dict3 = adam(x_initial=-4, y_initial=-4, total_iterations=1000, step_size=0.003, loss_function="booth",
                        beta_1=0.9, beta_2=0.999, eps=0.00001)
    x3 = return_dict3["x_all"]
    y3 = return_dict3["y_all"]
    print(3, x1[-1])
    print(3, y1[-1])
    plt.figure()
    plt.title("Adam optimiser on Booth function, step size = 0.003")
    plot_booth(x3, y3)

    # adam vs adagrad loss on booth
    plt.figure()
    return_dict4 = adam(x_initial=-4, y_initial=-4, total_iterations=1000, step_size=0.3, loss_function="booth",
                        beta_1=0.9, beta_2=0.999, eps=0.00001)
    loss1 = return_dict4["loss_all"]
    plot_loss(loss1)
    return_dict5 = ada_grad(x_initial=-4, y_initial=-4, total_iterations=1000, step_size=0.3, loss_function="booth")
    loss2 = return_dict5["loss_all"]
    plot_loss(loss2)
    plt.legend(["Adam", "AdaGrad"])
    plt.title("Adam and AdaGrad optimisers on Booth function")

    # adam vs adagrad loss on styblinski-tang
    plt.figure()
    return_dict6 = adam(x_initial=0, y_initial=0, total_iterations=90, step_size=0.3, loss_function="styblinski_tang",
                        beta_1=0.9, beta_2=0.999, eps=0.00001)
    loss3 = return_dict6["loss_all"]
    plot_loss(loss3)
    return_dict7 = ada_grad(x_initial=0, y_initial=0, total_iterations=90, step_size=0.3,
                            loss_function="styblinski_tang")
    loss4 = return_dict7["loss_all"]
    plot_loss(loss4)
    plt.legend(["Adam", "AdaGrad"])
    plt.title("Adam and AdaGrad optimisers on Styblinski-Tang function")

    # plot surfaces for visual in report.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_mesh_styblinski_tang, y_mesh_styblinski_tang, styblinski_matrix, cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    plt.title("Styblinski-Tang function")

    # plot surfaces for visual in report.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_mesh_booth, y_mesh_booth, booth_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title("Booth function")

    # draw plots
    plt.show()


run_demo()
