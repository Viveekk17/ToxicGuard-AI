from utils.prediction import predict_comment

comment = "You are an absolutely disgusting person."

results = predict_comment(comment)

print("\nTOXICGUARD AI – ANALYSIS RESULTS\n")

for result in results:

    print(
        f"{result['category']}: "
        f"{result['probability']:.4f} | "
        f"Detected: {result['detected']}"
    )