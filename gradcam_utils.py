import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv_layer"):

    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def overlay_heatmap(img, heatmap):

    # Ensure image is uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype("uint8")

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply color map → becomes 3-channel
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure both are same size
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

    # Overlay
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed