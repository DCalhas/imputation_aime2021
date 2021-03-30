from nilearn import plotting, image

def plot_view_mask(img, timestep=4, vmin=None, vmax=None, resampling_factor=4, symmetric_cmap=False, save_file="/tmp/plot.html"):
    img = image.index_img(img, timestep)

    if(vmin is None):
    	vmin=np.amin(img.get_data())
    if(vmax is None):
    	vmax=np.amax(img.get_data())

    view = plotting.view_img(img, 
    						threshold=None,
    						colorbar=True,
                            annotate=False,
                            draw_cross=False,
                            cut_coords=[0, 0,  0],
                            black_bg=True,
                            bg_img=False,
                            cmap="inferno",
                            symmetric_cmap=symmetric_cmap,
                            vmax=vmax,
                            vmin=vmin,
                            dim=-2,
                            resampling_interpolation="nearest",
                            title="Dataset 01 Downsampling " + str(resampling_factor))

    view.save_as_html(save_file)