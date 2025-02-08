import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from typing import List, Tuple
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Palette:
    def __init__(self, palette: List[any]):
        self.palette = palette

    def __getitem__(self, idx):
        return self.palette[idx % len(self.palette)]
    

MRK_PLT = Palette(['o', 'x', 's', 'D', 'v', '|', '+', '*', '^', '<', '>', 'p', 'h', 'H', 'X', '_'])
CLR_PLT = Palette(['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'navy', 'olive', 'silver', 'tan', 'aqua', 'fuchsia', 'lime', 'teal', 'indigo', 'maroon', 'navy', 'olive', 'silver', 'tan', 'aqua', 'fuchsia'])
    
class TableCell:
    def __init__(self, 
                 text, 
                 colors, 
                 text_color='contrast', 
                 text_background='white', 
                 text_alpha=0.5, 
                 text_edgecolor='white', 
                 text_fontsize=12):
        
        self.text = text
        self.colors = colors

        self.text_color = text_color
        self.text_background = text_background
        self.text_alpha = text_alpha
        self.text_edgecolor = text_edgecolor
        self.text_fontsize = text_fontsize



def plt_table(ax, cells: List[TableCell], cols, spacing=0.0):

    if cols == 'auto':
        cols = np.floor(np.sqrt(1.6 * len(cells)))

    cols = int(cols)

    spacing = 0 if spacing is None or spacing < 0 else spacing / 10

    rows = len(cells) // cols
    cell_width = 2
    cell_height = 1
    table_height = rows * cell_height + (rows - 1) * spacing
    table_width = cols * cell_width + (cols - 1) * spacing

    for r in range(rows):
        for c in range(cols):
            x = c * cell_width + c * spacing
            y = r * cell_height + r * spacing
            w = cell_width 
            h = cell_height
            cell = cells[r*cols + c]

            # divide the cell into n subcells and color each subcell
            for i, color in enumerate(cell.colors):
                ax.add_patch(mpatches.Rectangle((x, y + i * h / len(cell.colors)), w, h / len(cell.colors), color=color))

            # color = cells[r*cols + c].colors[0]
            # ax.add_patch(mpatches.Rectangle((x, y), w, h, color=color))

            text_x = x + w / 2
            text_y = y + h / 2


            text_color = cell.text_color
            if cell.text_color == 'contrast':
                rgb = matplotlib.colors.to_rgb(cell.text_background)
                luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                text_color = 'black' if luminance > 0.5 else 'white'
                

            ax.text(text_x, text_y, cell.text, color=text_color, fontsize=cell.text_fontsize, ha='center', va='center', 
                    bbox={'facecolor': cell.text_background, 'alpha': cell.text_alpha, 'edgecolor': cell.text_edgecolor})


    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

def plot_cube_by_origin(ax, center, size, color):
    half_size = size / 2
    vertices = np.array([[center[0] - half_size, center[1] - half_size, center[2] - half_size],
                         [center[0] + half_size, center[1] - half_size, center[2] - half_size],
                         [center[0] + half_size, center[1] + half_size, center[2] - half_size],
                         [center[0] - half_size, center[1] + half_size, center[2] - half_size],
                         [center[0] - half_size, center[1] - half_size, center[2] + half_size],
                         [center[0] + half_size, center[1] - half_size, center[2] + half_size],
                         [center[0] + half_size, center[1] + half_size, center[2] + half_size],
                         [center[0] - half_size, center[1] + half_size, center[2] + half_size]])

    faces = [[vertices[0], vertices[1], vertices[5], vertices[4]],
             [vertices[7], vertices[6], vertices[2], vertices[3]],
             [vertices[0], vertices[4], vertices[7], vertices[3]],
             [vertices[1], vertices[5], vertices[6], vertices[2]],
             [vertices[4], vertices[5], vertices[6], vertices[7]],
             [vertices[0], vertices[1], vertices[2], vertices[3]]]
    
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='r', alpha=0.1))


def plot_cube_by_vertex(ax, x1: Tuple[float, float, float], x2: Tuple[float, float, float], color,
                        alpha=0.1, edgecolor='k'
                        ):
    assert type(x1) == tuple and type(x2) == tuple, "x1 and x2 must be tuples"
    assert len(x1) == 3 and len(x2) == 3, "x1 and x2 must have 3 elements"

    # Find min and max for each coordinate
    x_min, x_max = min(x1[0], x2[0]), max(x1[0], x2[0])
    y_min, y_max = min(x1[1], x2[1]), max(x1[1], x2[1])
    z_min, z_max = min(x1[2], x2[2]), max(x1[2], x2[2])

    # Define the 8 vertices of the cube
    vertices = np.array([
        [x_min, y_min, z_min], [x_min, y_min, z_max],
        [x_min, y_max, z_min], [x_min, y_max, z_max],
        [x_max, y_min, z_min], [x_max, y_min, z_max],
        [x_max, y_max, z_min], [x_max, y_max, z_max]
    ])

    # Define the 6 faces of the cube using the vertex indices
    faces = [
        [vertices[i] for i in [0, 1, 3, 2]],  # Left face
        [vertices[i] for i in [4, 5, 7, 6]],  # Right face
        [vertices[i] for i in [0, 1, 5, 4]],  # Bottom face
        [vertices[i] for i in [2, 3, 7, 6]],  # Top face
        [vertices[i] for i in [0, 2, 6, 4]],  # Front face
        [vertices[i] for i in [1, 3, 7, 5]],  # Back face
    ]

    # Create and add the cube's faces to the plot
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edgecolor)
    ax.add_collection3d(poly3d)


if __name__ == "__main__":
    matplotlib.use('Gtk3Agg')

    fig, ax = plt.subplots(figsize=(10, 10))


    def icolor(color1, color2, factor):
        return [c1 + (c2 - c1) * factor for c1, c2 in zip(color1, color2)]

    ncells = 30
    cells = [TableCell(f'{i}', [
                            icolor((1, 0, 0), (0, 0, 1), i / ncells),
                            icolor((1, 0, 0), (0, 0, 1), (ncells - i) / ncells)
                                ]) for i in range(ncells)]
    plt_table(cells, 6, spacing=1)
    plt.show()