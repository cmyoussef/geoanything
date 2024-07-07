from . import gizmos


def populate_toolbar(toolbar, *args):
    try:
        gizmos.NukeToolbarGenerator(*args).populate_toolbar(toolbar)
    except Exception as e:
        print(f"Error populating toolbar: {e}")

