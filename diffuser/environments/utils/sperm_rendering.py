import math
import os

import numpy as np

from diffuser.environments.utils.Bezier import Bezier

import matplotlib.pyplot as plt
import json
import ast
import cv2
import matplotlib as mpl

## [ batch_size x horizon x observation_dim ]
def sperm_traj_render(data):
    output_img = np.zeros((400, 200*data.shape[0], 3), dtype=np.uint8)
    i = 0
    for d in data:
        tail = np.reshape(d[:40], (20, 2))
        params = np.reshape(d[40:50], (5, 2))
        angle = d[-3]
        head = data[:i+1, -2:]

        rot = mpl.transforms.Affine2D().rotate_deg(angle * 180)

        rot_tail = rot.transform(tail)
        displ = tail[-1] - rot_tail[-1]
        rot_tail += displ

        rot_params = rot.transform(params)
        displ2 = tail[-1] - rot_params[-1]
        rot_params += displ2

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        img = plt2opencv((rot_tail + 1) * 70, (model+ 1) * 70, (140, 140), (200, 200), parameters=((rot_params + 1) * 70))

        img2 = plt2opencvcoords(head, rot_tail * 70, resize=(200, 200))

        output_img[:200, i*200:i*200+200, :] = img
        output_img[200:, i*200:i*200+200, :] = img2

        i += 1

    return output_img

def vec2angle(vector, normalize=True):
    # angle
    m = (vector[1] + 1e-5) / (vector[0] + 1e-5)
    angle = math.atan(m)
    deg = math.degrees(angle)

    rand_noise = (np.random.rand() * 2 - 1)
    if np.isclose(vector[0], 0., atol=1e-5):
        if vector[1] > 0:
            deg = 90. + rand_noise
        else:
            deg = -90 + rand_noise
    elif np.isclose(vector[1], 0., atol=1e-5):
        if vector[0] > 0:
            deg = 0. + rand_noise
        else:
            deg = 180 + rand_noise
            if deg > 180.:
                deg = deg - 360
    else:
        if vector[0] < 0 and vector[1] > 0:  # If x < 0 and y > 0
            deg = 90 + (90 + deg)  # 90 +(90-|alpha|)
        elif vector[0] <=0 and vector[1] < 0:  # If x < 0 and y < 0
            deg = 180 + deg  # 180+alpha
        elif vector[0] > 0 and vector[1] < 0:  # If x > 0 and y < 0
            deg = 360 + deg  # 360-|alpha|

    if normalize:
        deg = (((deg / 180) + 1) % 2) - 1

    return deg


def sperm_traj_simp_render(data):
    output_img = np.zeros((400, 200*data.shape[0], 3), dtype=np.uint8)
    i = 0
    init_head = np.expand_dims((np.random.rand(2)*2)-1, axis=0)

    head = init_head
    for d in data:
        params = np.reshape(d[:10], (5, 2))
        velocity = d[-4:-2]
        correction_angle_vector = d[-2:]

        velocity_angle = vec2angle(velocity, normalize=False)
        correction_angle = vec2angle(correction_angle_vector, normalize=False)

        # a = velocity_angle - correction_angle
        # value = (np.radians(a) + np.pi) % (2*np.pi) - np.pi
        # if value > np.pi/3:
        #     print('Diff angles: ', np.degrees(value), velocity_angle, correction_angle)

        # angle = velocity_angle - correction_angle
        angle = correction_angle
        rot = mpl.transforms.Affine2D().rotate_deg(angle)
        rot_params = rot.transform(params)

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        img = plt2opencv((model + 1) * 70, img_size=(140, 140), resize=(200, 200), parameters=((rot_params + 1) * 70),
                         angle_vector=correction_angle_vector, direction_vector=velocity)

        img2 = plt2opencvcoords(head, model * 70, resize=(200, 200))

        head[:, 0] = head[:, 0] + ((velocity[0] / 1280) * 2)
        head[:, 1] = head[:, 1] + ((velocity[1] / 1024) * 2)

        output_img[:200, i*200:i*200+200, :] = img
        output_img[200:, i*200:i*200+200, :] = img2

        i += 1

    return output_img

def _ema(pred):
    alpha = 0.4
    ema = np.zeros((pred.shape[0],), dtype=np.float32)

    ema[0] = pred[0]

    for i in range(1, pred.shape[0]):
        ema[i] = (alpha * pred[i] + (1 - alpha) * ema[i-1])

    return ema

def sperm_traj_render_w_displacementEMA(data):
    output_img = np.zeros((400, 200*data.shape[0], 3), dtype=np.uint8)
    i = 0
    first_iter = True

    ema_angles = _ema(data[:, -3])

    for d, angle in zip(data, ema_angles):
        tail = np.reshape(d[:40], (20, 2))
        params = np.reshape(d[40:50], (5, 2))
        head = data[:i+1, -2:]


        rot = mpl.transforms.Affine2D().rotate_deg(angle * 180)

        rot_tail = rot.transform(tail)
        displ = tail[-1] - rot_tail[-1]
        rot_tail += displ

        rot_params = rot.transform(params)
        displ2 = tail[-1] - rot_params[-1]
        rot_params += displ2

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        img = plt2opencv((rot_tail + 1) * 70, (model+ 1) * 70, (140, 140), (200, 200), parameters=((rot_params + 1) * 70))

        if first_iter:
            first_iter = False
        else:
            head[i, 0] = head[i-1, 0] + ((displacement[0] * 20/ 1280)*2)
            head[i, 1] = head[i-1, 1] + ((displacement[1] * 20/ 1024)*2)

        displacement = data[i, -5:-3]
        img2 = plt2opencvcoords(head, rot_tail * 70, resize=(200, 200))

        output_img[:200, i*200:i*200+200, :] = img
        output_img[200:, i*200:i*200+200, :] = img2

        i += 1

    return output_img

def sperm_traj_render_w_displacement(data):
    output_img = np.zeros((400, 200*data.shape[0], 3), dtype=np.uint8)
    i = 0
    first_iter = True
    for d in data:
        tail = np.reshape(d[:40], (20, 2))
        params = np.reshape(d[40:50], (5, 2))
        angle = d[-3]
        head = data[:i+1, -2:]


        rot = mpl.transforms.Affine2D().rotate_deg(angle * 180)

        rot_tail = rot.transform(tail)
        displ = tail[-1] - rot_tail[-1]
        rot_tail += displ

        rot_params = rot.transform(params)
        displ2 = tail[-1] - rot_params[-1]
        rot_params += displ2

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        img = plt2opencv((rot_tail + 1) * 70, (model+ 1) * 70, (140, 140), (200, 200), parameters=((rot_params + 1) * 70))

        if first_iter:
            first_iter = False
        else:
            head[i, 0] = head[i-1, 0] + ((displacement[0] * 20/ 1280)*2)
            head[i, 1] = head[i-1, 1] + ((displacement[1] * 20/ 1024)*2)

        displacement = data[i, -5:-3]
        img2 = plt2opencvcoords(head, rot_tail * 70, resize=(200, 200))

        output_img[:200, i*200:i*200+200, :] = img
        output_img[200:, i*200:i*200+200, :] = img2

        i += 1

    return output_img


def sperm_traj_render_full_image_debug(data, dim=(1024, 1280),  resize=(300, 300), old_coords=None, linewidth=2,  head_size=7.0, recorded_positions=[]):
    record_positions = []

    i = 0

    plt.style.use('dark_background')
    plt.figure()

    for d in data:
        tail = np.reshape(d[:40], (20, 2))
        params = np.reshape(d[40:50], (5, 2))
        angle = d[-3]
        head = d[-2:]
        displacement = d[-5:-3]

        rot = mpl.transforms.Affine2D().rotate_deg(angle * 180)

        rot_tail = rot.transform(tail)
        displ = tail[-1] - rot_tail[-1]
        rot_tail += displ

        rot_params = rot.transform(params)
        displ2 = tail[-1] - rot_params[-1]
        rot_params += displ2

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        # img = plt2opencv((rot_tail + 1) * 70, (model+ 1) * 70, (140, 140), (200, 200), parameters=((rot_params + 1) * 70))

        rot_tail = rot_tail * 70
        model = model * 70
        rot_params = rot_params * 70

        record_positions.append(head)

        if old_coords is not None:
            head[0] = old_coords[i, 0] + ((displacement[0] * 20/ 1280)*2)
            head[1] = old_coords[i, 1] + ((displacement[1] * 20/ 1024)*2)

        x = ((head[0] + 1.) / 2.) * dim[1]
        y = ((head[1] + 1.) / 2.) * dim[0]

        # Convert angle to radians
        angle_radians = np.deg2rad(angle * 180)
        # Magnitude of the vector (change this to your desired magnitude)
        magnitude = 50.0
        # Calculate the components of the vector
        x_component = magnitude * np.cos(angle_radians)
        y_component = magnitude * np.sin(angle_radians)

        plt.quiver(x, y, x_component, y_component, angles='xy', scale_units='xy', scale=1, color='r', label='Vector', width=0.003) # headwidth=1, headlength=2)

        plt.plot(rot_tail[:, 0] + x,  # x-coordinates.
                 rot_tail[:, 1] + y, '-w', linewidth=linewidth)  # y-coordinates.
        circle = plt.Circle((rot_tail[-1, 0] + x, rot_tail[-1, 1] + y), head_size, color='w')

        plt.plot(model[:, 0] + x,  # x-coordinates.
                 model[:, 1] + y, '-g', linewidth=linewidth//2)  # y-coordinates.
        circle2 = plt.Circle((model[-1, 0] + x, model[-1, 1] + y), head_size/2, color='g')

        # plt.plot(
        #     rot_params[:, 0] + x,  # x-coordinates.
        #     rot_params[:, 1] + y,  # y-coordinates.
        #     'ro:',  # Styling (red, circles, dotted).
        #     linewidth=1
        # )

        recorded_positions[i].append([x, y])

        plt.plot(np.asarray(recorded_positions[i])[:, 0],  # x-coordinates.
                 np.asarray(recorded_positions[i])[:, 1], '-.b', linewidth=1)  # y-coordinates.
        circle = plt.Circle((rot_tail[-1, 0] + x, rot_tail[-1, 1] + y), head_size, color='w')

        plt.plot(x, y, '-b', linewidth=1)  # y-coordinates.
        ax = plt.gca()
        ax.add_patch(circle)
        ax.add_patch(circle2)

        plt.xlim([0, dim[1]])
        plt.ylim([0, dim[0]])

        i += 1

    #plt.axis('off')
    plt.grid()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2RGB)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()

    return graph_rgb, np.asarray(record_positions), recorded_positions

def sperm_traj_render_full_image_debug(data, dim=(1024, 1280),  resize=(300, 300), old_coords=None, linewidth=2,  head_size=7.0, recorded_positions=[], plot_id=False):
    record_positions = []

    i = 0

    plt.style.use('dark_background')
    plt.figure()

    for d in data:
        tail = np.reshape(d[:40], (20, 2))
        params = np.reshape(d[40:50], (5, 2))
        angle = d[-3]
        head = d[-2:]
        displacement = d[-5:-3]

        rot = mpl.transforms.Affine2D().rotate_deg(angle * 180)

        rot_tail = rot.transform(tail)
        displ = tail[-1] - rot_tail[-1]
        rot_tail += displ

        rot_params = rot.transform(params)
        displ2 = tail[-1] - rot_params[-1]
        rot_params += displ2

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        # img = plt2opencv((rot_tail + 1) * 70, (model+ 1) * 70, (140, 140), (200, 200), parameters=((rot_params + 1) * 70))

        rot_tail = rot_tail * 70
        model = model * 70
        rot_params = rot_params * 70

        record_positions.append(head)

        if old_coords is not None:
            head[0] = old_coords[i, 0] + ((displacement[0] * 20/ 1280)*2)
            head[1] = old_coords[i, 1] + ((displacement[1] * 20/ 1024)*2)

        x = ((head[0] + 1.) / 2.) * dim[1]
        y = ((head[1] + 1.) / 2.) * dim[0]

        # Convert angle to radians
        angle_radians = np.deg2rad(angle * 180)
        # Magnitude of the vector (change this to your desired magnitude)
        magnitude = 50.0
        # Calculate the components of the vector
        x_component = magnitude * np.cos(angle_radians)
        y_component = magnitude * np.sin(angle_radians)

        plt.quiver(x, y, x_component, y_component, angles='xy', scale_units='xy', scale=1, color='r', label='Vector', width=0.003) # headwidth=1, headlength=2)

        plt.plot(rot_tail[:, 0] + x,  # x-coordinates.
                 rot_tail[:, 1] + y, '-w', linewidth=linewidth)  # y-coordinates.
        circle = plt.Circle((rot_tail[-1, 0] + x, rot_tail[-1, 1] + y), head_size, color='w')

        plt.plot(model[:, 0] + x,  # x-coordinates.
                 model[:, 1] + y, '-g', linewidth=linewidth//2)  # y-coordinates.
        circle2 = plt.Circle((model[-1, 0] + x, model[-1, 1] + y), head_size/2, color='g')

        # plt.plot(
        #     rot_params[:, 0] + x,  # x-coordinates.
        #     rot_params[:, 1] + y,  # y-coordinates.
        #     'ro:',  # Styling (red, circles, dotted).
        #     linewidth=1
        # )

        recorded_positions[i].append([x, y])

        plt.plot(np.asarray(recorded_positions[i])[:, 0],  # x-coordinates.
                 np.asarray(recorded_positions[i])[:, 1], '-.b', linewidth=1)  # y-coordinates.
        circle = plt.Circle((rot_tail[-1, 0] + x, rot_tail[-1, 1] + y), head_size, color='w')

        plt.plot(x, y, '-b', linewidth=1)  # y-coordinates.

        if plot_id:
            plt.text(x+5, y+5, str(i), fontsize='x-small')

        ax = plt.gca()
        ax.add_patch(circle)
        ax.add_patch(circle2)

        plt.xlim([0, dim[1]])
        plt.ylim([0, dim[0]])

        i += 1

    #plt.axis('off')
    plt.grid()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2RGB)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()

    return graph_rgb, np.asarray(record_positions), recorded_positions


def sperm_traj_render_full_image(data, dim=(1024, 1280),  resize=(300, 300), old_coords=None, linewidth=2,  head_size=7.0):
    record_positions = []

    i = 0

    plt.style.use('dark_background')
    plt.figure()

    for d in data:
        tail = np.reshape(d[:40], (20, 2))
        params = np.reshape(d[40:50], (5, 2))
        angle = d[-3]
        head = d[-2:]
        displacement = d[-5:-3]


        rot = mpl.transforms.Affine2D().rotate_deg(angle * 180)

        rot_tail = rot.transform(tail)
        displ = tail[-1] - rot_tail[-1]
        rot_tail += displ

        rot_params = rot.transform(params)
        displ2 = tail[-1] - rot_params[-1]
        rot_params += displ2

        linspace = np.linspace(0., 1., num=20)
        model = Bezier.Curve(linspace, rot_params)

        # img = plt2opencv((rot_tail + 1) * 70, (model+ 1) * 70, (140, 140), (200, 200), parameters=((rot_params + 1) * 70))

        rot_tail = rot_tail * 70
        model = model * 70
        rot_params = rot_params * 70

        record_positions.append(head)

        if old_coords is not None:
            head[0] = old_coords[i, 0] + ((displacement[0] * 20/ 1280)*2)
            head[1] = old_coords[i, 1] + ((displacement[1] * 20/ 1024)*2)


        x = ((head[0] + 1.) / 2.) * dim[1]
        y = ((head[1] + 1.) / 2.) * dim[0]

        plt.plot(model[:, 0] + x,  # x-coordinates.
                 model[:, 1] + y, '-w', linewidth=linewidth)  # y-coordinates.
        circle = plt.Circle((model[-1, 0] + x, model[-1, 1] + y), head_size, color='w')

        ax = plt.gca()
        ax.add_patch(circle)

        plt.xlim([0, dim[1]])
        plt.ylim([0, dim[0]])

        i += 1

    plt.axis('off')
    #plt.grid()
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2RGB)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()

    return graph_rgb, np.asarray(record_positions)


def read_json(path):
    with open(path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    json_object = ast.literal_eval(json_object)
    return json_object


def read_frames(frame_path):
    frames = []
    for path, directories, files in os.walk(frame_path):
        files.sort()
        for f in files:
            if f.endswith('.jpg'):
                frames.append(plt.imread(os.path.join(path, f))[:, :140, :])
    return frames


def read_trajectory(spline_path, frames_path=None, num=25, shift=3):
    state = []
    coord = []
    head = []
    angle = []
    param = []
    for path, directories, files in os.walk(spline_path):
        files.sort()
        for f in files:
            if f.endswith('.json'):
                json_data = read_json(os.path.join(path, f))
                _parameters = np.asarray(json_data['spline_params'])

                linspace = np.linspace(0., 1., num=20)
                _model = Bezier.Curve(linspace, _parameters)

                _ravel_point_pairs = np.ravel(_model)
                _angle = json_data['correction_angle'] / 180.
                _head = json_data['head_coordinates']
                _head[0] = (_head[0] / 1024) * 2 - 1
                _head[1] = (1 - (_head[1] / 1280)) * 2 - 1
                _head[2] = _head[2] / len(files)
                state.append(np.concatenate([_ravel_point_pairs, [_angle], _head], axis=0))
                coord.append((_model / 70.) - 1)
                head.append(_head)
                angle.append(_angle)
                param.append((_parameters / 70.) - 1)

    if frames_path is not None:
        frames = read_frames(frames_path)
        curr_frames = []
        next_frames = []
    else:
        curr_frames = None
        next_frames = None
    curr_states = []
    curr_coords = []
    curr_heads = []
    curr_angles = []
    curr_params = []
    next_coords = []
    next_heads = []
    next_angles = []
    next_params = []

    shift = shift
    for i in range(len(state) - 1):
        j = np.maximum(i - 1, 0)
        k = np.maximum(i - 2, 0)
        curr_coords.append(coord[i])
        curr_heads.append(head[i])
        curr_angles.append(angle[i])
        curr_params.append(param[i])
        if frames_path is not None:
            curr_frames.append(frames[i])
        if j != i and k != i and k != j:
            curr_states.append(np.concatenate([state[i], state[j][:-1]]))
            next_coords.append(coord[i + 1])
            next_heads.append(head[i + 1])
            next_angles.append(angle[i + 1])
            next_params.append(param[i + 1])
            if frames_path is not None:
                next_frames.append(frames[i + 1])

    return curr_states, curr_coords, curr_heads, curr_angles, curr_params, curr_frames, next_coords, next_heads, next_angles, next_params, next_frames, shift


def plt2opencv(coordinates, real_coords=None, img_size=None, resize=None, linewidth=5, parameters=None, angle_vector=None, direction_vector=None):
    plt.style.use('dark_background')
    plt.figure()
    plt.plot(coordinates[:, 0],  # x-coordinates.
             coordinates[:, 1], '-w', linewidth=linewidth)  # y-coordinates.
    circle = plt.Circle((coordinates[-1, 0], coordinates[-1, 1]), 4.0, color='w')

    if real_coords is not None:
        plt.plot(real_coords[:, 0],  # x-coordinates.
                 real_coords[:, 1], '.-g', linewidth=linewidth-2)  # y-coordinates.
        circle2 = plt.Circle((real_coords[-1, 0], real_coords[-1, 1]), 3.0, color='g')

    if parameters is not None:
        plt.plot(
            parameters[:, 0],  # x-coordinates.
            parameters[:, 1],  # y-coordinates.
            'ro:'  # Styling (red, circles, dotted).
        )
    ax = plt.gca()
    ax.add_patch(circle)
    if real_coords is not None:
        ax.add_patch(circle2)

    if angle_vector is not None:
        magnitude = 30.0
        # Calculate the components of the vector
        x_component = magnitude * angle_vector[0]
        y_component = magnitude * angle_vector[1]
        plt.quiver(parameters[-1, 0], parameters[-1, 1], x_component, y_component, angles='xy', scale_units='xy', scale=1, color='r',
                   label='Vector', width=0.05)  # headwidth=1, headlength=2)

    if direction_vector is not None:
        magnitude = 10.0
        # Calculate the components of the vector
        x_component = magnitude * direction_vector[0]
        y_component = magnitude * direction_vector[1]
        plt.quiver(parameters[-1, 0], parameters[-1, 1], x_component, y_component, angles='xy', scale_units='xy', scale=1, color='b',
                   label='Vector', width=0.05)  # headwidth=1, headlength=2)

    plt.xlim([0, img_size[1]])
    plt.ylim([0, img_size[0]])
    # plt.axis('off')
    plt.grid()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2RGB)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()
    return graph_rgb



def plt2opencvcoords(xy, coordinates, img_size=(1024, 1280), resize=None, linewidth=2, parameters=None, head_size=7.0):
    x = ((xy[:, 0] + 1.) / 2.) * img_size[1]
    y = ((xy[:, 1] + 1.) / 2.) * img_size[0]

    plt.style.use('dark_background')
    plt.figure()
    plt.plot(coordinates[:, 0] + x[-1],  # x-coordinates.
             coordinates[:, 1] + y[-1], '-w', linewidth=linewidth)  # y-coordinates.
    circle = plt.Circle((coordinates[-1, 0] + x[-1], coordinates[-1, 1] + y[-1]), head_size, color='w')
    if parameters is not None:
        plt.plot(
            parameters[:, 0] + x[-1],  # x-coordinates.
            parameters[:, 1] + y[-1],  # y-coordinates.
            'ro:'  # Styling (red, circles, dotted).
        )

    plt.plot(x, y, '-b', linewidth=1)  # y-coordinates.
    ax = plt.gca()
    ax.add_patch(circle)

    plt.xlim([0, img_size[1]])
    plt.ylim([0, img_size[0]])
    # plt.axis('off')
    plt.grid()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2RGB)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()
    return graph_rgb


def plt2opencvlist(coordinates, img_size, resize=None, linewidth=5, parameters=None):
    plt.style.use('dark_background')
    plt.figure()

    first = True
    alpha = 1.
    alpha_list = []
    for i in range(len(coordinates)):
        alpha_list.append(alpha)
        alpha *= 0.8
    alpha_list = reversed(alpha_list)
    for coord in coordinates:
        plt.plot(coord[:, 0],  # x-coordinates.
                 coord[:, 1], '-', linewidth=linewidth, alpha=next(alpha_list))  # y-coordinates.
        circle = plt.Circle((coord[-1, 0], coord[-1, 1]), 4.0)

        if parameters is not None:
            plt.plot(
                parameters[:, 0],  # x-coordinates.
                parameters[:, 1],  # y-coordinates.
                'ro:'  # Styling (red, circles, dotted).
            )
        ax = plt.gca()
        ax.add_patch(circle)

    plt.xlim([0, img_size[1]])
    plt.ylim([0, img_size[0]])
    # plt.axis('off')
    plt.grid()
    fig = plt.gcf()
    fig.canvas.draw()
    graph_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the RGBA array to RGB
    graph_rgb = cv2.cvtColor(graph_array, cv2.COLOR_RGBA2RGB)

    if resize is not None:
        graph_rgb = cv2.resize(graph_rgb, (resize[1], resize[0]))

    plt.close()
    return graph_rgb
