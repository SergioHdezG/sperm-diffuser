from spllib.bezier_splines import Bezier
import matplotlib.pyplot as plt
import cv2
import numpy as np

def bezierSpline2image(complete_points):
    t_points = np.arange(0, 1, 0.05)

    visualizations = []
    for spline in list(complete_points):
        visualizations.append(visualize_bezier(spline))

    return np.asarray(visualizations)

def visualize_bezier(initial_points):
    plt.figure()
    t_points = np.arange(0, 1, 0.01)
    curve = Bezier.Curve(t_points, initial_points.reshape((int(initial_points.shape[0] / 2), 2)))
    # plt.plot(curve[:, 0], curve[:, 1], '-m', linewidth=5)
    # plt.plot(initial_points.reshape((int(initial_points.shape[0]/2), 2))[:, 0],
    #          initial_points.reshape((int(initial_points.shape[0]/2), 2))[:, 1], 'bo', linewidth=5)

    plt.figure()
    plt.plot(
        curve[:, 0],  # x-coordinates.
        curve[:, 1]  # y-coordinates.
    )
    plt.plot(
        initial_points.reshape((int(initial_points.shape[0] / 2), 2))[:, 0],  # x-coordinates.
        initial_points.reshape((int(initial_points.shape[0] / 2), 2))[:, 1],  # y-coordinates.
        'ro:'  # Styling (red, circles, dotted).
    )
    plt.plot(
        initial_points.reshape((int(initial_points.shape[0] / 2), 2))[0, 0],  # x-coordinates.
        initial_points.reshape((int(initial_points.shape[0] / 2), 2))[0, 1],  # y-coordinates.
        'go'  # Styling (red, circles, dotted).
    )
    plt.plot(
        initial_points.reshape((int(initial_points.shape[0] / 2), 2))[-1, 0],  # x-coordinates.
        initial_points.reshape((int(initial_points.shape[0] / 2), 2))[-1, 1],  # y-coordinates.
        'bo'  # Styling (red, circles, dotted).
    )
    plt.grid()
    # plt.show()
    # plt.xlim([0, 140])
    # plt.ylim([0, 140])
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2BGR)

    graph_rgb = cv2.resize(graph_rgb, (300, 300))

    return graph_rgb

