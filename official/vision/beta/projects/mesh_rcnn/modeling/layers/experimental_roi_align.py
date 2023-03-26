import tensorflow as tf
"""
ROIAlign test function that returns a 4D tensor of size
[batch_size, num_rois, output_size[0], output_size[1]]
similarly to the PyTorch implementation.

Not sure if this works correctly yet, may be unnecessary.
"""
class ROIAlign(tf.keras.layers.Layer):
    def __init__(self, output_size: int = 12, spatial_scale: float = 1.0):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def call(self, inputs):
        x, rois = inputs

        batch_size, channels, height, width = tf.unstack(tf.shape(x))

        num_rois = tf.shape(rois)[0]

        rois = tf.cast(rois, tf.float32)

        # Apply spatial scale to ROI coordinates
        rois /= self.spatial_scale

        # Compute ROI pooling grid
        rois = tf.reshape(rois, [num_rois, 1, 1, 5])

        roi_grid = self.compute_roi_grid(rois, self.output_size)

        # Extract pixel values from input tensor using the ROI grid
        pixel_values = self.extract_pixel_values(x, roi_grid)

        # Compute the weighted sum of the pixel values
        cells_weights = self.compute_cells_weights(roi_grid)

        pooled_features = tf.reduce_sum(cells_weights * pixel_values, axis=[2, 3])

        # Reshape the pooled features tensor to match the output size
        pooled_features = tf.reshape(pooled_features, [batch_size, num_rois, -1])

        pooled_features = tf.reshape(pooled_features, [batch_size, num_rois, self.output_size[0], self.output_size[1]])

        return pooled_features

    def compute_roi_grid(self, rois, output_size):
        num_rois, _, _, _ = tf.unstack(tf.shape(rois))

        # Compute cell size and grid size
        cell_height = (rois[:, :, :, 4] - rois[:, :, :, 2]) / output_size[0]
        cell_width = (rois[:, :, :, 3] - rois[:, :, :, 1]) / output_size[1]

        grid_height = tf.cast(tf.math.ceil(cell_height), tf.int32)
        grid_width = tf.cast(tf.math.ceil(cell_width), tf.int32)

        # Compute ROI pooling grid
        grid_y = tf.range(output_size[0], dtype=tf.float32)
        grid_x = tf.range(output_size[1], dtype=tf.float32)

        grid_y = tf.expand_dims(grid_y, axis=0)
        grid_x = tf.expand_dims(grid_x, axis=0)

        grid_y = tf.tile(grid_y, [num_rois, output_size[1]])
        grid_x = tf.tile(grid_x, [num_rois, output_size[0]])

        rois_y1 = rois[:, :, :, 2]
        rois_x1 = rois[:, :, :, 1]

        roi_grid_y = (grid_y * tf.expand_dims(cell_height, axis=2) + tf.expand_dims(rois_y1, axis=2)) / tf.cast(height, tf.float32)
        roi_grid_x = (grid_x * tf.expand_dims(cell_width, axis=2) + tf.expand_dims(rois_x1, axis=2)) / tf.cast(width, tf.float32)

        roi_grid = tf.stack([tf.zeros_like(roi_grid_y), roi_grid_y, roi_grid_x, tf.ones_like(roi_grid_y)], axis=3)

        return roi_grid

    def extract_pixel_values(self, x, roi_grid):
        batch_size, channels, height, width = tf.unstack(tf.shape(x))

        num_rois = tf.shape(roi_grid)[0]

        # Reshape input tensor to 2D tensor
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.reshape(x, [batch_size * height * width, channels])

        # Compute indices of pixel values to extract
        indices = self.compute_indices(roi_grid, height, width)

        # Gather pixel values using indices
        pixel_values = tf.gather(x, indices)

        # Reshape the pixel values tensor to match the output size
        pixel_values = tf.reshape(pixel_values, [num_rois, -1, channels])

        return pixel_values

    def compute_indices(self, roi_grid, height, width):
        num_rois, grid_height, grid_width, _ = tf.unstack(tf.shape(roi_grid))

        # Convert ROI grid coordinates to pixel indices
        roi_grid = tf.reshape(roi_grid, [num_rois, -1, 4])
        roi_grid *= tf.constant([0, height, width, height, width], dtype=tf.float32)
        roi_grid = tf.cast(roi_grid, tf.int32)

        # Compute indices of pixel values to extract
        indices = roi_grid[:, :, 0] * width * height + roi_grid[:, :, 1] * width + roi_grid[:, :, 2]

        # Reshape indices tensor to match the output size
        indices = tf.reshape(indices, [num_rois, -1, 1])

        return indices

    def compute_cells_weights(self, roi_grid):
        num_rois, grid_height, grid_width, _ = tf.unstack(tf.shape(roi_grid))

        # Compute cell coordinates
        cell_y = tf.cast(tf.range(grid_height), tf.float32) + 0.5
        cell_x = tf.cast(tf.range(grid_width), tf.float32) + 0.5

        cell_y /= tf.cast(grid_height, tf.float32)
        cell_x /= tf.cast(grid_width, tf.float32)

        cell_y = tf.expand_dims(cell_y, axis=0)
        cell_x = tf.expand_dims(cell_x, axis=0)

        cell_y = tf.tile(cell_y, [num_rois, grid_width, 1])
        cell_x = tf.tile(cell_x, [num_rois, 1, grid_height])
        cell_x = tf.transpose(cell_x, [0, 2, 1])

        cell_coords = tf.stack([tf.zeros_like(cell_y), cell_y, cell_x, tf.ones_like(cell_y)], axis=3)

        # Compute cell weights
        cells_weights = self.compute_weights(roi_grid, cell_coords)

        return cells_weights

    def compute_weights(self, roi_grid, cell_coords):
        num_rois, grid_height, grid_width, _ = tf.unstack(tf.shape(roi_grid))

        # Compute weights for each ROI and cell
        weights = []
        for i in range(num_rois):
            roi = roi_grid[i, :, 0, :]
            roi = tf.expand_dims(roi, axis=0)
            cell_roi = tf.subtract(cell_coords, roi)
            cell_roi = tf.transpose(cell_roi, [0, 2, 1, 3])
            cell_roi = tf.reshape(cell_roi, [-1, 4])

            # Compute cell weights
            cell_weights = self.bilinear_interpolation_weights(cell_roi)

            cell_weights = tf.reshape(cell_weights, [grid_height, grid_width, -1])
            cell_weights = tf.transpose(cell_weights, [2, 0, 1])
            weights.append(cell_weights)

        weights = tf.stack(weights, axis=0)

        return weights

    def bilinear_interpolation_weights(self, coords):
        coords_y, coords_x = tf.unstack(coords, axis=1)
        # Compute the 4 points closest to the coordinates
        coords_y_floor = tf.floor(coords_y)
        coords_x_floor = tf.floor(coords_x)
        coords_y_ceil = coords_y_floor + 1
        coords_x_ceil = coords_x_floor + 1

        # Compute the distance between the coordinates and the 4 points
        dy_floor = coords_y - coords_y_floor
        dx_floor = coords_x - coords_x_floor
        dy_ceil = coords_y_ceil - coords_y
        dx_ceil = coords_x_ceil - coords_x

        # Compute the weights
        weights_floor = dy_ceil * dx_ceil
        weights_ceil = dy_floor * dx_floor
        weights_top_left = dy_floor * dx_ceil
        weights_bottom_right = dy_ceil * dx_floor

        # Stack the weights in the correct order
        weights = tf.stack([weights_top_left, weights_floor, weights_ceil, weights_bottom_right], axis=1)

        return weights


