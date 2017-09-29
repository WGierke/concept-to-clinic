from src.algorithms.segment.src.data_generation import prepare_training_data_cubes
from src.algorithms.segment.src.training import train


def test_3d_unet():
    prepare_training_data_cubes()
    train()


# def test_train_model():
#     ASSETS_DIR = '../assets/'
#     inputs = ["annotation_84_input.npy", "annotation_90_input.npy"]
#     outputs = ["annotation_84_output.npy", "annotation_90_output.npy"]
#
#     input_data = np.array((len(inputs), 64, 64, 64, 1))
#     output_data = np.array((len(inputs), 64, 64, 64, 1))
#
#     for index, in_file, out_file in enumerate(zip(inputs, outputs)):
#         input_cube = np.load(in_file)
#         output_cube = np.load(out_file)
#
#         # Expand dimensions
#         input_cube = np.expand_dims(input_cube, axis=0)
#         output_cube = np.expand_dims(output_cube, axis=0)
#         # Trailing channel is necessary for tensorflow
#         input_cube = input_cube.reshape(64, 64, 64, 1)
#         output_cube = output_cube.reshape(64, 64, 64, 1)
#
#         input_data[index] = input_cube
#         output_data[index] = output_cube
#
#
#     model = unet_model_3d((64, 64, 64, 1), downsize_filters_factor=5)
#     model.fit(input_data, output_data)
#
#     # # Predict
#     # predicted = model.predict(input_cube)
#     # np.save("predicted.npy", predicted)


def cube_show_slider(cube, axis=2, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    (Kudos to http://nbarbey.github.io/2011/07/08/matplotlib-slider.html)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    im = cube[s].squeeze()

    # display image
    l = ax.imshow(im, **kwargs)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
             for i in range(3)]
        im = cube[s].squeeze()
        l.set_data(im, **kwargs)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()

# def test_cube_segmentation_mask_crop():
#     """Test whether the annotations of the LIDC images are inside the segmented lung masks.
#     Iterate over all local LIDC images, fetch the annotations, compute their positions within the masks and check that
#     at this point the lung masks are set to 255."""
#
#     dicom_paths = get_dicom_paths()
#     path = dicom_paths[0]
#     min_z, max_z = get_z_range(path)
#     directories = path.split('/')
#     lidc_id = directories[2]
#     patient_id = directories[-1]
#     original_shape, mask_shape = save_lung_segments(path, patient_id)
#     scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == lidc_id).first()
#     annotation = scan.annotations[0]
#     centroid_x, centroid_y, centroid_z = annotation.centroid()
#     patient_image = load_patient_images(patient_id, wildcard="*_i.png", exclude_wildcards=[])
#     patient_mask = load_patient_images(patient_id, wildcard="*_m.png", exclude_wildcards=[])
#     # Normalize
#     patient_mask = np.divide(patient_mask, 255)
#     # Apply mask
#     lung_image = np.multiply(patient_image, patient_mask)
#     np.save("lung.npy", lung_image)
#
#     x_mask = int(mask_shape[1] / original_shape[1] * centroid_x)
#     y_mask = int(mask_shape[2] / original_shape[2] * centroid_y)
#     z_mask = int(abs(min_z) - abs(centroid_z))
#
#     annotation_cube = lung_image[z_mask-32:z_mask+32, x_mask-32:x_mask+32, y_mask-32:y_mask+32]
#     print(annotation_cube.shape)
#     print(type(annotation_cube))
#     np.save("cube.npy", annotation_cube)
#     assert False
