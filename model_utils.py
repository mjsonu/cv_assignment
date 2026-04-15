import numpy as np
import cv2

def preprocess_image(image):
    img = np.array(image)

    # Fix channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (128,128))
    img = img / 255.0

    return np.expand_dims(img, axis=0)


def predict(image, feature_extractor, pca, voting_clf, class_names):
    img_input = preprocess_image(image)

    features = feature_extractor.predict(img_input)
    features_pca = pca.transform(features)

    probs = voting_clf.predict_proba(features_pca)[0]

    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    return pred_class, confidence, probs, img_input