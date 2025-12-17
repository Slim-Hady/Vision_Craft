# application.py

from flask import Flask, render_template, request, jsonify
import io
import base64
from PIL import Image
import numpy as np

import image_ops as ops  # the file above

app = Flask(__name__)

def pil_png_base64(uint8_img):
    """
    Convert uint8 numpy image to base64 PNG string for HTML <img> src.
    """
    pil_img = Image.fromarray(uint8_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    operation = request.form.get('operation')

    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    img = ops.load_image(img_file)  # float [0,1]

    result = None
    extra = {}

    try:
        if operation == 'gaussian':
            mean = float(request.form.get('gauss_mean', 0.0))
            sigma = float(request.form.get('gauss_sigma', 0.05))
            result = ops.add_gaussian_noise(img, mean, sigma)

        elif operation == 'saltpepper':
            amount = float(request.form.get('sp_amount', 0.02))
            s_vs_p = float(request.form.get('sp_ratio', 0.5))
            result = ops.add_salt_pepper_noise(img, amount, s_vs_p)

        elif operation == 'median':
            size = int(request.form.get('median_size', 3))
            result = ops.apply_median_filter(img, size)

        elif operation == 'average':
            size = int(request.form.get('avg_size', 3))
            result = ops.apply_average_filter(img, size)

        elif operation == 'gamma':
            gamma = float(request.form.get('gamma', 1.0))
            result = ops.apply_gamma_correction(img, gamma)

        elif operation == 'grayscale':
            result = ops.to_grayscale(img)

        elif operation == 'histeq':
            # equalized image + histograms
            result = ops.apply_hist_equalization(img)
            orig_hist, orig_bins = ops.compute_histogram(img)
            enh_hist, enh_bins = ops.compute_histogram(result)
            extra['orig_hist'] = orig_hist
            extra['orig_bins'] = orig_bins
            extra['enh_hist'] = enh_hist
            extra['enh_bins'] = enh_bins

        elif operation == 'fusion_simple':
            if 'image2' not in request.files or request.files['image2'].filename == '':
                return jsonify({'error': 'Second image required'}), 400
            img2 = ops.load_image(request.files['image2'])
            result = ops.simple_average_fusion(img, img2)

        elif operation == 'fusion_weighted':
            if 'image2' not in request.files or request.files['image2'].filename == '':
                return jsonify({'error': 'Second image required'}), 400
            alpha = float(request.form.get('alpha', 0.5))
            img2 = ops.load_image(request.files['image2'])
            result = ops.weighted_fusion(img, img2, alpha)

        else:
            return jsonify({'error': 'Unknown operation'}), 400

    except Exception as e:
        return jsonify({'error': f'Processing failed: {e}'}), 500

    # Prepare images
    orig_uint8 = ops.to_uint8(img)
    if result.ndim == 2:  # grayscale -> stack to RGB for consistent display
        result_uint8 = ops.to_uint8(np.stack([result]*3, axis=2))
    else:
        result_uint8 = ops.to_uint8(result)

    return jsonify({
        'original': pil_png_base64(orig_uint8),
        'processed': pil_png_base64(result_uint8),
        **extra
    })

if __name__ == '__main__':
    app.run(debug=True)
