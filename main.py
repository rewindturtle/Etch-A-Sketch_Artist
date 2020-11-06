from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import networkx as nx
import Convert2Etch as c2e
import cv2
from scipy.spatial import distance as dst
import matplotlib.pyplot as plt
import threading


# The motors have trouble rotating when only moving one step at a time
# Moving two steps at a time seems to negate this issue
ETCH_SCALE_FACTOR = 4


def calculate_path(filename, rotation, shade, thresh1, thresh2, blur, sigma):
    global og_img_lock, num_nodes, num_subgraphs, global_x, global_y, load_bar_value, load_bar_txt, updating_lock, end_update
    end_update = False
    og_img_lock.acquire()
    get_original_display(filename,
                         rotation=-rotation)
    etch_nodes = convert_image(filename,
                               rotation=rotation,
                               shade=shade,
                               min_thresh=thresh1,
                               max_thresh=thresh2,
                               blur=blur,
                               sigma=sigma)
    og_img_lock.release()
    if end_update:
        updating_lock.release()
        return
    num_nodes = np.sum(etch_nodes).astype(int)
    graph = convert_nodes(etch_nodes)
    if end_update:
        updating_lock.release()
        return
    num_subgraphs = len(list(nx.connected_components(graph)))
    connected_graph = connect_subgraphs(graph)
    if end_update:
        updating_lock.release()
        return
    global_x, global_y = find_path(connected_graph)
    if end_update:
        updating_lock.release()
        return
    save_path_as_image(global_x, global_y)
    load_bar_txt = ""
    load_bar_value = 0
    updating_lock.release()


def get_original_display(filename, rotation=0):
    image_array = cv2.imread(filename)
    image_array = np.rot90(image_array, -rotation)
    image_height, image_width, channel = image_array.shape
    height_factor = 228 / image_height
    width_factor = 318 / image_width
    scale_factor = min(height_factor, width_factor)
    dim = (int(np.ceil(scale_factor * image_width)), int(np.ceil(scale_factor * image_height)))
    rs_im = cv2.resize(image_array, dim)
    cv2.imwrite('images/original.png', rs_im)


# Converts an image into an array the size of the etch-a-sketch screen where 1 represents an edge
def convert_image(image, height=620,
                  width=864,
                  rotation=0,
                  scale_before_canny=True,
                  min_thresh=50,
                  max_thresh=100,
                  shade=False,
                  blur=5,
                  sigma=75):

    global nodes_img_lock
    height = int(height / ETCH_SCALE_FACTOR)
    width = int(width / ETCH_SCALE_FACTOR)

    image_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # save the image as a 2 dimensional array (no colour channels)
    image_array = np.rot90(image_array, rotation)
    image_height, image_width = image_array.shape
    height_factor = height / image_height
    width_factor = width / image_width
    scale_factor = min(height_factor, width_factor)
    if scale_factor == width_factor:
        dim = (width, int(np.floor(scale_factor * image_height)))
    else:
        dim = (int(np.floor(scale_factor * image_width)), height)

    if scale_before_canny:
        image_array = cv2.resize(image_array, dim, interpolation=cv2.INTER_AREA)
    im_copy = image_array.copy() # saves a copy to do shading later
    image_array = cv2.bilateralFilter(image_array, blur, sigma, sigma)
    image_array = cv2.Canny(image_array, min_thresh, max_thresh)
    if not scale_before_canny:
        image_array = cv2.resize(image_array, dim, interpolation=cv2.INTER_AREA)
        im_copy = cv2.resize(im_copy, dim, interpolation=cv2.INTER_AREA)

    image_y, image_x = image_array.shape
    image_nodes = np.zeros(image_array.shape)
    image_nodes[image_array > 0] = 1

    if shade:
        im_p1 = np.ones(image_array.shape)
        im_p2 = np.ones(image_array.shape)
        im_p3 = np.ones(image_array.shape)

        diag_1 = np.zeros(image_array.shape)
        diag_2 = np.zeros(image_array.shape)
        diag_3 = np.zeros(image_array.shape)

        im_p1[im_copy >= 192] = 0
        im_p1[im_copy < 128] = 0
        im_p2[im_copy >= 128] = 0
        im_p2[im_copy < 64] = 0
        im_p3[im_copy >= 64] = 0

        for i in range(-image_y, image_x, 512):
            diag_1 += np.eye(image_y, image_x, k=i)
        for i in range(-image_y, image_x, 64):
            diag_2 += np.eye(image_y, image_x, k=i)
        for i in range(-image_y, image_x, 8):
            diag_3 += np.eye(image_y, image_x, k=i)

        image_nodes += (im_p1 * diag_1) + (im_p2 * diag_2) + (im_p3 * diag_3)

    x_min = int((width - image_x) / 2)
    x_max = width - image_x - x_min
    y_min = int((height - image_y) / 2)
    y_max = height - image_y - y_min

    if y_min > 0:
        image_nodes = np.concatenate((image_nodes, np.zeros((y_min, width)).astype(int)), axis=0)
        image_nodes = np.concatenate((np.zeros((y_max, width)).astype(int), image_nodes), axis=0)
    else:
        image_nodes = np.concatenate((image_nodes, np.zeros((height, x_min)).astype(int)), axis=1)
        image_nodes = np.concatenate((np.zeros((height, x_max)).astype(int), image_nodes), axis=1)
    rs_nodes = 255 * (1 - image_nodes)
    rs_nodes = cv2.resize(rs_nodes, (318, 228), interpolation=cv2.INTER_NEAREST)
    nodes_img_lock.acquire()
    cv2.imwrite('images/nodes.png', rs_nodes)
    nodes_img_lock.release()
    return image_nodes


# converts an array that indicates where the edges of an image are to a graph
def convert_nodes(nodes):
    global load_bar_value, load_bar_txt
    load_bar_txt = "Creating Graph"
    total_nodes = 100 * nodes.shape[0]
    image_graph = nx.Graph()
    image_graph.add_node((0, 0))
    for i in range(0, nodes.shape[0]):
        for j in range(0, nodes.shape[1]):
            if nodes[i][j] == 1:
                if i < nodes.shape[0] - 1:
                    if nodes[i + 1][j] == 1:
                        image_graph.add_edge((i, j), (i + 1, j))
                if j < nodes.shape[1] - 1:
                    if nodes[i][j + 1] == 1:
                        image_graph.add_edge((i,j), (i, j + 1))
                if i < nodes.shape[0] - 1 and j < nodes.shape[1] - 1:
                    if nodes[i + 1][j + 1] == 1:
                        image_graph.add_edge((i, j), (i + 1, j + 1))
                if i < nodes.shape[0] - 1 and j > 0:
                    if nodes[i + 1][j - 1] == 1:
                        image_graph.add_edge((i, j), (i + 1, j - 1))
            if end_update:
                return image_graph
        load_bar_value = (i + 1) * total_nodes
    return image_graph


# finds the closest pair of nodes between to unconnected graphs
def find_closest_nodes(source_nodes, target_nodes):
    distances = dst.cdist(source_nodes, target_nodes, 'euclidean')
    min_distances = np.min(distances, axis=1)
    arg_min_distances = np.argmin(distances, axis=1)
    arg_min_dist = np.argmin(min_distances)
    min_source = source_nodes[arg_min_dist]
    min_target = target_nodes[arg_min_distances[arg_min_dist]]
    return min_source, min_target


# creates a path of nodes that connects two unconnected nodes
def bridge_nodes(graph, source, target):
    delta_x = target[1] - source[1]
    delta_y = target[0] - source[0]
    grid = nx.Graph()
    for i in range(0, abs(delta_y) + 1):
        for j in range(0, abs(delta_x) + 2):
            if i < abs(delta_y):
                grid.add_edge((i, j), (i + 1, j), weight=1)
            if j < abs(delta_x):
                grid.add_edge((i, j), (i, j + 1))
            if i < abs(delta_y) and j < abs(delta_x):
                grid.add_edge((i, j), (i + 1, j + 1))
            if i < abs(delta_y) and j > 0:
                grid.add_edge((i, j), (i + 1, j - 1))
    bridge = list(nx.bidirectional_shortest_path(grid, (0, 0), (abs(delta_y), abs(delta_x))))
    x1 = source[1]
    y1 = source[0]
    for i in range(1, len(bridge)):
        x2 = source[1] + np.sign(delta_x) * bridge[i][1]
        y2 = source[0] + np.sign(delta_y) * bridge[i][0]
        graph.add_edge((y1, x1), (y2, x2))
        x1 = x2
        y1 = y2
    return graph


# Connects all subgraphs into one graph
def connect_subgraphs(graph):
    global load_bar_value, load_bar_txt
    load_bar_txt = "Connecting Subgraphs"
    load_bar_value = 0
    subgraphs = sorted(list(nx.connected_components(graph)), key=len)
    total_subgraphs = 100 / len(subgraphs)
    i = 1
    while len(subgraphs) > 1:
        min_subgraph = list(subgraphs[0])
        r_graph = graph.copy()
        for r_node in min_subgraph:
            r_graph.remove_node(r_node)
        outer_nodes = list(nx.nodes(r_graph))
        del r_graph
        min_source, min_target = find_closest_nodes(min_subgraph, outer_nodes)
        graph = bridge_nodes(graph, min_source, min_target)
        subgraphs = sorted(list(nx.connected_components(graph)), key=len)
        if end_update:
            return graph
        load_bar_value = i * total_subgraphs
        i += 1
    load_bar_value = 100
    return graph


# Finds the shortest path that visits all nodes
def find_path(graph):
    global load_bar_value, load_bar_txt
    load_bar_txt = "Calculating Path"
    load_bar_value = 0
    path = list(nx.dfs_preorder_nodes(graph, (0, 0)))
    nodes_y = [node[0] for node in path]
    nodes_x = [node[1] for node in path]
    total_nodes = 100 / len(nodes_y)
    x = [0]
    y = [0]
    for i in range(1, len(nodes_y)):
        if abs(nodes_x[i] - nodes_x[i - 1]) > 1 or abs(nodes_y[i] - nodes_y[i - 1]) > 1:
            bridge = list(nx.bidirectional_shortest_path(graph, (nodes_y[i - 1], nodes_x[i - 1]), (nodes_y[i], nodes_x[i])))
            bridge_y = [node[0] for node in bridge]
            bridge_x = [node[1] for node in bridge]
            for j in range(0, len(bridge) - 1):
                x.append(bridge_x[j])
                y.append(-bridge_y[j])
        else:
            x.append(nodes_x[i])
            y.append(-nodes_y[i])
        if end_update:
            return x, y
        load_bar_value = i * total_nodes
    return x, y


# Saves the path as an image
def save_path_as_image(x, y):
    global path_img_lock
    plt.cla()
    height = int(620 / ETCH_SCALE_FACTOR)
    width = int(864 / ETCH_SCALE_FACTOR)
    plt.plot(x, y, color='black')
    plt.xlim(0, width)
    plt.ylim(-height, 0)
    plt.axis('off')
    path_img_lock.acquire()
    plt.savefig('images/path.png', bbox_inches='tight', pad_inches = 0)
    if len(plt.get_fignums()) > 1:
        plt.close()
    image_array = cv2.imread('images/path.png')
    image_array[image_array < 245] = 0
    image_array[image_array >= 245] = 255
    image_array = cv2.resize(image_array, (318, 228), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('images/path.png', image_array)
    path_img_lock.release()


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        # True == Draw, False == Cancel
        self._draw_button_state = True
        self._cancel_drawing = False

        self.title("Etch A Sketch Artist")
        self.minsize(1024, 600)
        self.filename = "images/turtle.png"

        self.orientation = 0
        self.orientation_list = [0, 90, 180, -90]
        self.orientation_idx = 0
        self.orientation_txt = StringVar()
        self.orientation_txt.set("0°")

        self.shade_txt = StringVar()
        self.shade_value = False
        self.shade_txt.set("OFF")

        self.blur = 2
        self.blur_txt = StringVar()
        self.blur_txt.set("5")
        self.sigma = 75
        self.sigma_txt = StringVar()
        self.sigma_txt.set("75")

        self.thresh1 = 50
        self.thresh2 = 100
        self.thresh1_txt = StringVar()
        self.thresh1_txt.set("50")
        self.thresh2_txt = StringVar()
        self.thresh2_txt.set("100")

        self.browse_label_frame = ttk.LabelFrame(self,
                                                 text="Select Image")
        self.browse_label_frame.grid(column=0,
                                     row=0,
                                     padx=20,
                                     pady=20)

        self.filler_label = ttk.Label(self,
                                      text="",
                                      width=55)
        self.filler_label.grid(column=0,
                               row=2,
                               padx=20,
                               pady=20)

        self.draw_label_frame = ttk.LabelFrame(self,
                                               text="Draw Image")
        self.draw_label_frame.grid(column=0,
                                   row=1,
                                   padx=20,
                                   pady=20)

        self.rotate_label_frame = ttk.LabelFrame(self,
                                                 text="Orientation")
        self.rotate_label_frame.grid(column=1,
                                     row=0,
                                     padx=20,
                                     pady=20)

        self.shade_label_frame = ttk.LabelFrame(self,
                                                text="Shade")
        self.shade_label_frame.grid(column=1,
                                    row=1,
                                    padx=20,
                                    pady=20)

        self.thresh_label_frame = ttk.LabelFrame(self,
                                                 text="Edge Detection Thresholds")
        self.thresh_label_frame.grid(column=2,
                                     row=1,
                                     rowspan=1,
                                     padx=20,
                                     pady=20)

        self.blur_label_frame = ttk.LabelFrame(self,
                                               text="Bilateral Filter Kernel and Co-ordinate Space Size")
        self.blur_label_frame.grid(column=2,
                                   row=0,
                                   padx=20,
                                   pady=20)

        self.og_image_frame = ttk.LabelFrame(self,
                                             text="Original",
                                             width=159,
                                             height=114)
        self.og_image_frame .grid(column=0,
                                  row=2,
                                  padx=20,
                                  pady=20)

        self.node_image_frame = ttk.LabelFrame(self,
                                               text="Nodes",
                                               width=159,
                                               height=114)
        self.node_image_frame.grid(column=1,
                                   row=2,
                                   padx=20,
                                   pady=20)

        self.path_image_frame = ttk.LabelFrame(self,
                                               text="Path",
                                               width=159,
                                               height=114)
        self.path_image_frame.grid(column=2,
                                   row=2,
                                   padx=20,
                                   pady=20)
        self.nodes_frame = ttk.LabelFrame(self,
                                          text="Number of Nodes",
                                          width=159,
                                          height=57)
        self.nodes_frame.grid(column=0,
                              row=3,
                              padx=20,
                              pady=20)
        self.subgraph_frame = ttk.LabelFrame(self,
                                             text="Number of Subgraphs",
                                             width=159,
                                             height=57)
        self.subgraph_frame.grid(column=1,
                                 row=3,
                                 padx=20,
                                 pady=20)
        self.steps_frame = ttk.LabelFrame(self,
                                          text="Number of Steps",
                                          width=159,
                                          height=57)
        self.steps_frame.grid(column=2,
                              row=3,
                              padx=20,
                              pady=20)
        self.load_bar_frame = ttk.LabelFrame(self,
                                             text="",
                                             width=159,
                                             height=57)
        self.load_bar_frame.grid(column=1,
                                 row=4,
                                 padx=20,
                                 pady=20)

        self.browse_button()
        self.draw_button()
        self.rotate_button()
        self.shade_button()
        self.blur_slider()
        self.thresh()
        self.og_image()
        self.node_image()
        self.path_image()
        self.nodes_label()
        self.subgraphs_label()
        self.steps_label()
        self.progress_bar()
        self._update()

    def update_ui(self):
        global load_bar_value, og_img_lock, num_nodes, nodes_img_lock, load_bar_txt, num_subgraphs, global_x, global_y, path_img_lock
        self.load_bar_label['text'] = load_bar_txt
        self.load_bar['value'] = load_bar_value
        og_img_lock.acquire()
        self.og_img_src = ImageTk.PhotoImage(Image.open("images/original.png"))
        self.og_img.configure(image=self.og_img_src)
        self.og_img.image = self.og_img_src
        og_img_lock.release()
        self.num_nodes_label['text'] = "{}".format(num_nodes)
        nodes_img_lock.acquire()
        self.nodes_img_src = ImageTk.PhotoImage(Image.open("images/nodes.png"))
        self.nodes_img.configure(image=self.nodes_img_src)
        self.nodes_img.image = self.nodes_img_src
        nodes_img_lock.release()
        self.num_subgraphs_label['text'] = "{}".format(num_subgraphs)
        self.x, self.y = global_x, global_y
        self.num_steps_label['text'] = "{}".format(len(self.x))
        path_img_lock.acquire()
        self.path_img_src = ImageTk.PhotoImage(Image.open("images/path.png"))
        self.path_img.configure(image=self.path_img_src)
        self.path_img.image = self.path_img_src
        path_img_lock.release()
        root.after(50, self.update_ui)

    def browse_button(self):
        self.button = ttk.Button(self.browse_label_frame,
                                 text="Browse",
                                 command=self.fileDialog)
        self.button.grid(column=0,
                         row=0)

    def draw_button(self):
        self.motor_button = ttk.Button(self.draw_label_frame,
                                       text="Draw",
                                       command=self.draw)
        self.motor_button.grid(column=0,
                               row=0)

    def rotate_button(self):
        self.orientation_button = ttk.Button(self.rotate_label_frame,
                                             textvariable=self.orientation_txt,
                                             command=self.rotate)
        self.orientation_button.grid(column=0,
                                     row=0)

    def shade_button(self):
        self.fill_button = ttk.Button(self.shade_label_frame,
                                      textvariable=self.shade_txt,
                                      command=self.shade)
        self.fill_button.grid(column=0,
                              row=0)

    def blur_slider(self):
        self.blur_slider_1 = ttk.Scale(self.blur_label_frame,
                                       from_=0,
                                       to=15,
                                       orient=HORIZONTAL,
                                       command=self.update_blur_txt)
        self.blur_slider_1.set(self.blur)
        self.blur_slider_1.grid(column=0,
                                row=0)
        self.sigma_slider = ttk.Scale(self.blur_label_frame,
                                      from_=1,
                                      to=200,
                                      orient=HORIZONTAL,
                                      command=self.update_sigma_txt)
        self.sigma_slider.set(self.sigma)
        self.sigma_slider.grid(column=0,
                               row=1)

        self.blur_label = Label(self.blur_label_frame,
                                textvariable=self.blur_txt)
        self.blur_label.grid(column=1,
                             row=0)
        self.sigma_label = Label(self.blur_label_frame,
                                 textvariable=self.sigma_txt)
        self.sigma_label.grid(column=1,
                              row=1)

        self.blur_button = ttk.Button(self.blur_label_frame,
                                      text="Set",
                                      command=self.set_blur)
        self.blur_button.grid(column=2,
                              row=0,
                              rowspan=2)

    def thresh(self):
        self.thresh_slider_1 = ttk.Scale(self.thresh_label_frame,
                                         from_=0,
                                         to=300,
                                         orient=HORIZONTAL,
                                         command=self.update_thresh1_txt)
        self.thresh_slider_1.set(self.thresh1)
        self.thresh_slider_1.grid(column=0,
                                  row=0)
        self.thresh_slider_2 = ttk.Scale(self.thresh_label_frame,
                                         from_=0,
                                         to=300,
                                         orient=HORIZONTAL,
                                         command=self.update_thresh2_txt)
        self.thresh_slider_2.set(self.thresh2)
        self.thresh_slider_2.grid(column=0,
                                  row=1)
        self.thresh_label_1 = Label(self.thresh_label_frame,
                                    textvariable=self.thresh1_txt)
        self.thresh_label_1.grid(column=1,
                                 row=0)
        self.thresh_label_2 = Label(self.thresh_label_frame,
                                    textvariable=self.thresh2_txt)
        self.thresh_label_2.grid(column=1,
                                 row=1)
        self.thresh_button = ttk.Button(self.thresh_label_frame,
                                      text="Set",
                                      command=self.set_thresh)
        self.thresh_button.grid(column=2,
                                row=0,
                                rowspan=2)

    def progress_bar(self):
        self.load_bar = ttk.Progressbar(self.load_bar_frame,
                                        orient=HORIZONTAL,
                                        length=500,
                                        mode='determinate')
        self.load_bar.grid(column=0,
                           row=1)
        self.load_bar['value'] = 0
        self.load_bar_label = Label(self.load_bar_frame)
        self.load_bar_label['text'] = ""
        self.load_bar_label.grid(column=0,
                                 row=0)

    def og_image(self):
        self.og_img_src = ImageTk.PhotoImage(Image.open("images/turtle.png"))
        self.og_img = ttk.Label(self.og_image_frame,
                                image=self.og_img_src)
        self.og_img.grid(column=0,
                         row=0)

    def node_image(self):
        self.nodes_img_src = ImageTk.PhotoImage(Image.open("images/turtle_nodes.png"))
        self.nodes_img = ttk.Label(self.node_image_frame,
                                   image=self.nodes_img_src)
        self.nodes_img.grid(column=0,
                            row=0)

    def path_image(self):
        self.path_img_src = ImageTk.PhotoImage(Image.open("images/turtle_path.png"))
        self.path_img = ttk.Label(self.path_image_frame,
                                  image=self.path_img_src)
        self.path_img.grid(column=1,
                           row=1)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir="C:/Users/natha/Pictures",
                                                   title="Select A File",
                                                   filetype=(("picture files", "*.jpg *.png *.jpeg *.jfif *.tiff"),
                                                             ("all files", "*.*")))
        try:
            self._update()
        except:
            t = 0

    def rotate(self):
        self.orientation_idx = (self.orientation_idx + 1) % 4
        self.orientation = self.orientation_list[self.orientation_idx]
        self.orientation_txt.set("{}°".format(self.orientation))
        self._update()

    def shade(self):
        self.shade_value = not self.shade_value
        if self.shade_value:
            self.shade_txt.set("ON")
        else:
            self.shade_txt.set("OFF")
        self._update()

    def update_blur_txt(self, value):
        self.blur = int(float(value))
        self.blur_txt.set(str(2 * self.blur + 1))

    def update_sigma_txt(self, value):
        self.sigma = int(float(value))
        self.sigma_txt.set(str(self.sigma))

    def update_thresh1_txt(self, value):
        self.thresh1 = int(float(value))
        self.thresh1_txt.set(str(self.thresh1))

    def update_thresh2_txt(self, value):
        self.thresh2 = int(float(value))
        self.thresh2_txt.set(str(self.thresh2))

    def set_thresh(self):
        self._update()

    def set_blur(self):
        self._update()

    def _update(self):
        global updating_lock, end_update
        end_update = True
        updating_lock.acquire()
        t = threading.Thread(target=calculate_path,
                             args=(self.filename,
                                   -self.orientation_idx,
                                   self.shade_value,
                                   self.thresh1,
                                   self.thresh2,
                                   2 * self.blur + 1,
                                   self.sigma,),
                             daemon=True)
        t.start()

    def nodes_label(self):
        self.num_nodes_label = ttk.Label(self.nodes_frame,
                                         text="Test",
                                         justify="center")
        self.num_nodes_label.grid(column=1,
                                  row=1)

    def subgraphs_label(self):
        self.num_subgraphs_label = ttk.Label(self.subgraph_frame,
                                             text="Test",
                                             justify="center")
        self.num_subgraphs_label.grid(column=1,
                                      row=1)

    def steps_label(self):
        self.num_steps_label = ttk.Label(self.steps_frame,
                                         text="Test",
                                         justify="center")
        self.num_steps_label.grid(column=1,
                                  row=1)

    def draw(self):
        if self._draw_button_state:
            m_x, m_y = c2e.get_motor_movements(self.x, self.y)
            self.motor_button.configure(text="Cancel")
            print("Drawing")
            self._draw_button_state = False
        else:
            self.motor_button.configure(text="Draw")
            print("Cancelled")
            self._draw_button_state = True


if __name__ == '__main__':
    # global variables
    load_bar_value = 0
    num_nodes = 0
    load_bar_txt = ""
    global_x, global_y = [], []
    end_update = False
    og_img_lock = threading.Lock()
    nodes_img_lock = threading.Lock()
    path_img_lock = threading.Lock()
    updating_lock = threading.Lock()


    root = Root()
    root.after(0, root.update_ui)
    root.mainloop()
    root.quit()