from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from Count_Calculation import analyze_count, load_count_model
from Coverage_Calculation import (
    analyze_image,
    load_segmentation_model,
)
from Park_Health import generate_park_assessment


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}


def decode_uploaded_image(uploaded_file):
    """Validate an uploaded image and return an RGB NumPy array."""
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image uploads must be 20 MiB or smaller.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError(
                    "Unsupported image format. Upload a JPEG, PNG, BMP, or WebP."
                )
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "The image is too large to process (maximum 40 megapixels)."
                )
            image.load()
            return np.asarray(image.convert("RGB")).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "The uploaded file is not a valid, readable image."
        ) from error


@st.cache_resource(show_spinner=False)
def get_segmentation_model():
    return load_segmentation_model()


@st.cache_resource(show_spinner=False)
def get_count_model():
    return load_count_model()


def render_assessment(assessment):
    if assessment.get("error"):
        st.warning(
            "The numerical heuristic scores are available, but the AI narrative "
            f"could not be generated: {assessment['error']}"
        )

    score_columns = st.columns(2)
    score_columns[0].metric(
        "Overall park health",
        f"{assessment['overall_park_health']:.2f} / 100",
    )
    score_columns[1].metric(
        "Lawn quality",
        f"{assessment['lawn_quality']:.2f} / 100",
    )
    st.caption(assessment["scoring_methodology"])

    st.subheader("Assessment")
    for title, field in (
        ("Overall", "overall_assessment"),
        ("Lawn", "lawn_assessment"),
        ("Vegetation", "vegetation_assessment"),
        ("Cleanliness", "cleanliness_assessment"),
        ("Infrastructure", "infrastructure_assessment"),
    ):
        st.markdown(f"**{title}**")
        st.write(assessment[field])

    st.markdown("**Key observations**")
    if assessment["key_observations"]:
        for observation in assessment["key_observations"]:
            st.markdown(f"- {observation}")
    else:
        st.write("No AI-generated observations are available.")

    st.markdown("**Maintenance recommendations**")
    if assessment["maintenance_recommendations"]:
        for recommendation in assessment["maintenance_recommendations"]:
            st.markdown(f"- {recommendation}")
    else:
        st.write("No AI-generated recommendations are available.")

    st.markdown("**Assessment limitations**")
    for limitation in assessment["limitations"]:
        st.markdown(f"- {limitation}")


def main():
    st.set_page_config(
        page_title="Park Surveillance Assessment",
        page_icon="🌳",
        layout="wide",
    )
    st.title("Park Surveillance & Health Assessment")
    st.write(
        "Analyze one park or drone image for land-cover coverage, visible "
        "infrastructure counts, and a cautious, evidence-based assessment."
    )

    with st.sidebar:
        st.header("Local models")
        st.write("Segmentation: `models/best.pt`")
        st.write("Object detection: `models/Count.pt`")
        st.caption(
            "The analysis uses the supplied image only; it is not a park-wide "
            "survey or a diagnostic inspection."
        )

    uploaded_file = st.file_uploader(
        "Upload a park or drone image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPEG, PNG, BMP, and WebP; maximum 20 MiB.",
    )
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        image = decode_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Original image")
    st.image(image, use_container_width=True)
    if not st.button("Analyze image", type="primary"):
        return

    try:
        with st.spinner("Loading models and analyzing the image..."):
            segmentation_model = get_segmentation_model()
            coverage_report, segmentation_image, coverage_metrics = (
                analyze_image(image, segmentation_model)
            )

            count_model = get_count_model()
            count_report, detection_image, object_counts = analyze_count(
                image, count_model
            )

            assessment = generate_park_assessment(
                coverage_metrics,
                object_counts,
            )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        st.error(f"Analysis could not be completed: {error}")
        return
    except Exception as error:
        st.error(f"An unexpected analysis error occurred: {error}")
        st.exception(error)
        return

    st.subheader("Annotated images")
    segmentation_column, detection_column = st.columns(2)
    segmentation_column.image(
        cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB),
        caption="Segmentation",
        use_container_width=True,
    )
    detection_column.image(
        cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB),
        caption="Object detection",
        use_container_width=True,
    )

    st.subheader("Coverage analysis")
    st.table(
        [
            {"Metric": metric, "Coverage (%)": percentage}
            for metric, percentage in coverage_metrics.items()
        ]
    )
    with st.expander("Segmentation report"):
        st.markdown(coverage_report)

    st.subheader("Object counting")
    st.table(
        [
            {"Object": class_name, "Count": count}
            for class_name, count in object_counts.items()
        ]
    )
    with st.expander("Object detection report"):
        st.markdown(count_report)

    st.subheader("Park health assessment")
    render_assessment(assessment)


if __name__ == "__main__":
    main()
