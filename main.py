import math
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

CATEGORIES = ['a', 'i', 'u',
              'ga', 'gi', 'gu',
              'ka', 'ki', 'ku',
              'nga', 'ngi', 'ngu',
              'ta', 'ti', 'tu',
              'da', 'di', 'du',
              'na', 'ni', 'nu',
              'la', 'li', 'lu',
              'sa', 'si', 'su',
              'ma', 'mi', 'mu',
              'pa', 'pi', 'pu',
              'ba', 'bi', 'bu',
              'wi', 'wu',
              'yi', 'yu'
              ]

CONSONANTS = [
    'ga', 'gi', 'gu',
    'ka', 'ki', 'ku',
    'nga', 'ngi', 'ngu',
    'ta', 'ti', 'tu',
    'da', 'di', 'du',
    'na', 'ni', 'nu',
    'la', 'li', 'lu',
    'sa', 'si', 'su',
    'ma', 'mi', 'mu',
    'pa', 'pi', 'pu',
    'ba', 'bi', 'bu',
    'wi', 'wu',
    'yi', 'yu'
]


loaded_model = tf.keras.models.load_model('kulitan_recognition_cnn.h5')


def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadow_removed = cv2.merge(result_norm_planes)
    return shadow_removed


def remove_noise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)


def joinNeighborContours(contours, merge_area_padding, garlit_size_thresh):
    original_contours = []
    box_contours = []
    for i in range(len(contours[0])):
        x, y, w, h = cv2.boundingRect(contours[0][i])
        original_contours.append([x, y, w, h])
        max_size = max(w, h)
        if (max_size == w):
            box_x = x
            y_center = math.ceil(y+(h/2))
            box_y = y_center - math.ceil(max_size/2)
            box_contours.append([box_x, box_y, max_size, max_size])
        else:
            box_y = y
            x_center = math.ceil(x+(w/2))
            box_x = x_center - math.ceil(max_size/2)
            box_contours.append([box_x, box_y, max_size, max_size])

    contours_sum = 0
    for i in range(len(box_contours)):
        contours_sum += (w*h)

    contours_avg_size = contours_sum/len(box_contours)
    contours_merged = []
    merged = []

    for i in range(len(box_contours)):
        if i in merged:
            continue
        padding = math.ceil(box_contours[i][3] * merge_area_padding)
        x = original_contours[i][0]
        y = original_contours[i][1]
        w = original_contours[i][2]
        h = original_contours[i][3]
        x_end = x + w
        y_end = y + h
        if (box_contours[i][2]*box_contours[i][3] < contours_avg_size * garlit_size_thresh):
            continue
        ixp = box_contours[i][0]
        iyp = box_contours[i][1] - padding
        iwp = box_contours[i][2]
        ihp = box_contours[i][3] + (padding * 2)
        ixp_end = ixp + iwp
        iyp_end = iyp + ihp
        icenter = [math.ceil(ixp+(iwp/2)), math.ceil(iyp+(ihp/2))]
        for j in range(len(original_contours)):
            jx = original_contours[j][0]
            jy = original_contours[j][1]
            jw = original_contours[j][2]
            jh = original_contours[j][3]
            jx_end = jx + jw
            jy_end = jy + jh
            jcenter = [math.ceil(jx+(jw/2)), math.ceil(jy+(jh/2))]
            jxp = box_contours[j][0]
            jyp = box_contours[j][1] - padding
            jwp = box_contours[j][2]
            jhp = box_contours[j][3] + (padding * 2)
            jxp_end = jxp + jwp
            jyp_end = jyp + jhp
            if (i == j):
                continue
            else:
                if (w*h > jw*jh):
                    if ((ixp < jcenter[0] < (ixp+iwp)) and (iyp < jcenter[1] < (iyp+ihp))):
                        merged.append(j)
                        x = min(x, jx)
                        y = min(y, jy)
                        x_end = max(x_end, jx_end)
                        y_end = max(y_end, jy_end)
                        w = x_end - x
                        h = y_end - y
                        ixp = min(ixp, jxp)
                        iyp = min(iyp, jyp)
                        ixp_end = max(ixp_end, jxp_end)
                        iyp_end = max(iyp_end, jyp_end)
                        iwp = ixp_end - ixp
                        ihp = iyp_end - iyp
                        icenter = [math.ceil(x+(w/2)), math.ceil(y+(h/2))]
                else:
                    if ((jxp < icenter[0] < (jxp+jwp)) and (jyp < icenter[1] < (jyp+jhp))):
                        merged.append(j)
                        x = min(x, jx)
                        y = min(y, jy)
                        x_end = max(x_end, jx_end)
                        y_end = max(y_end, jy_end)
                        w = x_end - x
                        h = y_end - y
                        ixp = min(ixp, jxp)
                        iyp = min(iyp, jyp)
                        ixp_end = max(ixp_end, jxp_end)
                        iyp_end = max(iyp_end, jyp_end)
                        iwp = ixp_end - ixp
                        ihp = iyp_end - iyp
                        icenter = [math.ceil(x+(w/2)), math.ceil(y+(h/2))]
        contours_merged.append([x, y, w, h])
    return contours_merged


def sortContours(contours):
    by_y = sorted(contours, key=lambda x: x[1])
    line_y = by_y[0][1]
    line_h = by_y[0][3]
    by_line = []
    line = []
    for x, y, w, h in by_y:
        y_center = y + math.ceil(h/2)
        if y > line_y + line_h:
            by_line.append(line)
            line = []
            line_y = y
            line_h = h
        if (line_y < y_center < line_y+line_h):
            line.append((x, y, w, h))
    by_line.append(line)
    contours_grouped_sorted = []
    for i in range(len(by_line)):
        line = sorted(by_line[i], key=lambda x: x[0])
        contours_grouped_sorted.append(line)
    return contours_grouped_sorted


def extractImages(image, groupedContours):
    final_grouped_images = []
    line = 0
    for line_contours in groupedContours:
        images = []
        for i in range(len(line_contours)):
            images.append(image[line_contours[i][1]:line_contours[i][1]+line_contours[i]
                          [3], line_contours[i][0]:line_contours[i][0]+line_contours[i][2]])
        final_line_images = []
        for i in range(len(images)):
            h, w = images[i].shape
            min_dimension = max(w, h)
            max_axis = 'x' if max(w, h) == w else 'y'
            if max_axis == 'y':
                left = math.ceil((min_dimension - w)/2)
                right = (min_dimension - w) - left
                squaredImg = cv2.copyMakeBorder(
                    images[i].copy(), 0, 0, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            elif max_axis == 'x':
                top = math.ceil((min_dimension - h)/2)
                bottom = (min_dimension - h) - top
                squaredImg = cv2.copyMakeBorder(
                    images[i].copy(), top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            padding = math.ceil(min_dimension * 0.1)
            paddedImg = cv2.copyMakeBorder(squaredImg.copy(
            ), padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            final_line_images.append(paddedImg)
        final_grouped_images.append(final_line_images)
        line += 1
    return final_grouped_images


def processGroupedImages(groupedImages):
    processedGroupedImages = []
    for imageGroup in groupedImages:
        imgGroupProcessed = []
        for img in imageGroup:
            im = img.copy()
            inverted_img = (255 - im)
            thresh = cv2.threshold(inverted_img, 0, 255,
                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            resized_img = cv2.resize(thresh, (28, 28))
            imgGroupProcessed.append(resized_img)
        processedGroupedImages.append(imgGroupProcessed)
    return processedGroupedImages


def preprocess(image):
    invImage = (255 - image)
    blur = cv2.GaussianBlur(invImage, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_contours = joinNeighborContours(contours, 0.2, 0.3)
    sortedContours = sortContours(merged_contours)
    groupedImages = extractImages(image.copy(), sortedContours)
    processedGroupedImages = processGroupedImages(groupedImages)
    return [processedGroupedImages, sortedContours]


def predictImages(groupedImages):
    predictions_percentages = []
    predictions = []
    for imageGroup in groupedImages:
        groupedPercentages = []
        groupedPredictions = []
        for imgdata in imageGroup:
            img_array = np.expand_dims(imgdata, axis=0)
            img_channel = img_array.reshape(img_array.shape[0], 28, 28, 1)
            normalized = img_channel / 255
            prediction = loaded_model.predict([normalized])
            one_hot = np.argmax(prediction, axis=1)
            groupedPercentages.append(prediction[0][one_hot][0])
            groupedPredictions.append(CATEGORIES[one_hot[0]])
        predictions.append(groupedPredictions)
        predictions_percentages.append(groupedPercentages)
    return [predictions, predictions_percentages]


def illustratePredictions(predictions, groupedContours, image):
    bounded_img = image.copy()
    for i in range(len(groupedContours)):
        for j in range(len(groupedContours[i])):
            x = groupedContours[i][j][0] * 3
            y = groupedContours[i][j][1] * 3
            w = groupedContours[i][j][2] * 3
            h = groupedContours[i][j][3] * 3
            label = predictions[0][i][j]
            percentage = round(predictions[1][i][j] * 100, 2)
            cv2.rectangle(bounded_img, (x, y), (x+w, y+h), (0, 0, 0), 1)
            cv2.putText(bounded_img, label + " (" + str(percentage) + "%)",
                        (x, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)
    return bounded_img


def joinChars(predictedLabels):
    predictedJoined = []
    for i in range(len(predictedLabels)):
        lineJoin = ""
        for j in range(len(predictedLabels[i])):
            if j == (len(predictedLabels[i])-1) and len(predictedLabels[i]) > 1:
                if predictedLabels[i][j] in CONSONANTS:
                    chopped = predictedLabels[i][j][:-1]
                    lineJoin += chopped
            else:
                lineJoin += predictedLabels[i][j]
        lineStr = lineJoin.replace("aa", "á").replace("ai", "e").replace("au", "o").replace(
            "ii", "í").replace("uu", "ú").replace("ia", "ya").replace("ua", "wa")
        predictedJoined.append(lineStr)
    finalString = ''.join(predictedJoined)
    return finalString


def driverCode(image):
    img = image.copy()
    resized = cv2.resize(img, (300, 400))
    resized_tri = cv2.resize(img, (900, 1200))
    image_removed_shadow = shadow_remove(resized)
    de_noised_img = remove_noise(image_removed_shadow)
    imgGrayscale = cv2.cvtColor(de_noised_img, cv2.COLOR_BGR2GRAY)
    preprocessed = preprocess(imgGrayscale)
    predictedLabels = predictImages(preprocessed[0])
    imagePredictions = illustratePredictions(
        predictedLabels, preprocessed[1], resized_tri)
    predictedWord = joinChars(predictedLabels[0])
    img = Image.fromarray(imagePredictions.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, format="JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode("utf-8")
    response = {}
    response["prediction"] = predictedWord
    response["image"] = str(img_base64)
    return response



@app.route("/", methods=['GET', "POST"])
def index():
    if request.method == "POST":
        image_file = request.files.get('file')
        if image_file is None or image_file.filename == "":
            return jsonify({"error": "no file"})
        try:
            image_bytes = image_file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            converted_img = pillow_img.convert("RGB")
            open_cv_image = np.array(converted_img)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            pred = driverCode(open_cv_image)
            return jsonify(pred)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))