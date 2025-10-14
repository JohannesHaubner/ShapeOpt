import pyvista as pv
import numpy as np
from colormaps import cmap_1, cmap_2

# specify output directory 
from pathlib import Path
here = Path(__file__).parent.parent.resolve()

def make_movie(foldername : str, gifname : str):

    # specify filename and read mesh    
    filename = str(here) + "/Output/Forward/" + foldername + "/characteristic_function.xdmf"
    filename2 = str(here) + "/Output/Forward/" + foldername + "/velocity.xdmf"

    reader = pv.get_reader(filename)
    reader2 = pv.get_reader(filename2)

    # filename for mp4
    gif_fn = gifname

    import os
    os.system("export RDMAV_FORK_SAFE=0")

    pv.start_xvfb()
    reader.set_active_time_point(0)
    reader2.set_active_time_point(0)
    data = []
    m = 3
    for i in range(m):
        data.append(reader.read())
        data[i].rotate_x(90, inplace=True)
    for j in range(2):
        data.append(reader2.read())
        data[m +j].rotate_x(90, inplace=True)
    pl = pv.Plotter(window_size=([1024, 768]),off_screen=True)
    

    #needs to be specified
    vmax = 1e-0
    n = 2
    pl.open_gif(gif_fn, fps=50/n)
    # plot velocity magnitude
    mag = np.sum(np.abs(data[m+0]["v"])**2, axis=-1)**(1./2)
    data[m+0].point_data.set_scalars(mag, name="magnitude")



    pl.add_mesh(data[m+0], cmap=cmap_1, clim=[0., vmax], opacity="linear")
    # plot channel with opacity
    pl.add_mesh(data[0], cmap=cmap_2, clim=[0., 1.], opacity=0.2, lighting=True, smooth_shading=True, split_sharp_edges=True)
    # remove opacity for elastic flap
    pl.add_mesh(data[1], cmap=cmap_2, clim=[0., 1.], opacity="linear", lighting=True, smooth_shading=True, split_sharp_edges=True)
    # add glyphs
    glyphs = data[m+1].glyph(orient="v", scale="v", factor=0.1)
    pl.add_mesh(glyphs, cmap=cmap_1, clim=[0., vmax])
    pl.show(auto_close=False)
    pl.set_scale(xscale=-1,yscale=-1)
    pl.enable_shadows()
    pl.remove_scalar_bar("magnitude")
    pl.remove_scalar_bar("GlyphScale")
    pl.remove_scalar_bar()
    pl.update()
    pl.write_frame()

    for i in range(int(len(reader.time_values)/n)):
            pl.clear_actors()
            reader.set_active_time_point(i*n)
            reader2.set_active_time_point(i*n)
            print('active time value:', reader.active_time_value)
            m = 3
            for i in range(m):
                data[i] = reader.read()
                data[i].rotate_x(90, inplace=True)
            for j in range(2):
                data[m+j] = reader2.read()
                data[m +j].rotate_x(90, inplace=True)
            # plot velocity magnitude
            mag = np.sum(np.abs(data[m+0]["v"])**2, axis=-1)**(1./2)
            data[m+0].point_data.set_scalars(mag, name="magnitude")
            pl.add_mesh(data[m+0], cmap=cmap_1, clim=[0., vmax], opacity="linear")
            # plot channel with opacity
            pl.add_mesh(data[0], cmap=cmap_2, clim=[0., 1.], opacity=0.2, lighting=True, smooth_shading=True, split_sharp_edges=True)
            # remove opacity for elastic flap
            pl.add_mesh(data[1], cmap=cmap_2, clim=[0., 1.], opacity="linear", lighting=True, smooth_shading=True, split_sharp_edges=True)
            # add glyphs
            glyphs = data[m+1].glyph(orient="v", scale="v", factor=0.1)
            pl.add_mesh(glyphs, cmap=cmap_1, clim=[0., vmax])
            pl.show(auto_close=False)
            pl.remove_scalar_bar("magnitude")
            pl.remove_scalar_bar("GlyphScale")
            pl.remove_scalar_bar()
            pl.update()
            pl.write_frame()
    pl.close()
    pass

if __name__ == "__main__":
    foldername = "3d_forward_2"
    outname = "test3d.gif"
    make_movie(foldername, outname)