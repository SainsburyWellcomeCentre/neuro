import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import napari
from napari.utils.io import magic_imread
from imlib.general.system import get_sorted_file_paths

from imlib.IO.cells import cells_xml_to_df, cells_to_xml
from imlib.cells.cells import Cell
from magicgui import magicgui


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="img_paths", type=str, help="Directory of images")
    parser.add_argument(
        dest="cells_xml", type=str, help="Path to the .xml cell file"
    )
    parser.add_argument(
        "--symbol", type=str, default="ring", help="Marker symbol."
    )
    parser.add_argument(
        "--marker-size", type=int, default=15, help="Marker size."
    )
    parser.add_argument(
        "--opacity", type=float, default=0.6, help="Opacity of the markers."
    )
    return parser


def napari_array_to_cell_list(cell_array, type=-1):
    cell_list = []
    for row in range(0, len(cell_array)):
        cell_list.append(Cell(np.flip(cell_array[row]), type))

    return cell_list


def napari_cells_to_xml(cells, non_cells, xml_file_path):
    cell_list = napari_array_to_cell_list(cells, type=Cell.CELL)
    non_cell_list = napari_array_to_cell_list(non_cells, type=Cell.UNKNOWN)

    all_cells = cell_list + non_cell_list
    cells_to_xml(all_cells, xml_file_path)


def cells_df_as_np(cells_df, new_order=[2, 1, 0], type_column="type"):
    cells_df = cells_df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()
    return cells


def get_cell_arrays(cells_file):
    df = cells_xml_to_df(cells_file)

    non_cells = df[df["type"] == Cell.UNKNOWN]
    cells = df[df["type"] == Cell.CELL]

    cells = cells_df_as_np(cells)
    non_cells = cells_df_as_np(non_cells)
    return cells, non_cells


def main():
    args = parser().parse_args()
    img_paths = get_sorted_file_paths(args.img_paths, file_extension=".tif")
    cells, non_cells = get_cell_arrays(args.cells_xml)

    with napari.gui_qt():
        v = napari.Viewer(title="Cellfinder cell viewer")
        images = magic_imread(img_paths, use_dask=True, stack=True)
        v.add_image(images)

        non_cell_layer = v.add_points(
            non_cells,
            size=args.marker_size,
            n_dimensional=True,
            opacity=args.opacity,
            symbol=args.symbol,
            face_color="lightskyblue",
            name="Non-Cells",
        )
        cell_layer = v.add_points(
            cells,
            size=args.marker_size,
            n_dimensional=True,
            opacity=args.opacity,
            symbol=args.symbol,
            face_color="lightgoldenrodyellow",
            name="Cells",
        )

        @magicgui(call_button="Save_cells")
        def save_cells():
            print("Saving cells")
            napari_cells_to_xml(
                cell_layer.data, non_cell_layer.data, args.cells_xml
            )
            print("Finished!")

        v.window.add_dock_widget(
            save_cells.Gui(), name="Save Cells", area="bottom"
        )


if __name__ == "__main__":
    main()
