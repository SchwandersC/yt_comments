from flask import Flask, request, render_template
from src.predict import run_video_prediction
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt_pipeline")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    video_id = ""

    if request.method == "POST":
        video_id = request.form.get("video_id", "").strip()

        if not video_id:
            error = "Please enter a YouTube video ID."
        else:
            try:
                prediction = run_video_prediction(video_id)
                logger.info(f"Prediction returned: {prediction}")
            except Exception as e:
                logger.exception("Prediction failed.")
                error = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error, video_id=video_id)

if __name__ == "__main__":
    app.run(debug=True)
