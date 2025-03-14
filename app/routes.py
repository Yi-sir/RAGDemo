import json

import requests
from flask import Flask, jsonify, render_template, request


def register_routes(app, retriever, generator):
    @app.route("/")
    def index():
        return render_template("index.html")


@app.route("/api/answer", methods=["POST"])
def get_answer():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    contexts = retriever.retrieve(question)
    answer = generator.generate(question, contexts)
    return jsonify({"answer": answer})
