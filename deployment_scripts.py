from ultralytics import YOLO
import cv2
import os

# Make sure this points to your TRAINED weights (e.g., runs/detect/train/weights/best.pt)
model = YOLO("runs/detect/train5/weights/best.pt")

if __name__ == "__main__":
    images_folder = input(
        "Enter the path to the folder containing images for inference: "
    )

    if not os.path.isdir(images_folder):
        print("Error: The provided path is not a valid directory.")
    else:
        print(f"Running inference on all images in: {images_folder}")

        # YOLO handles directory paths directly!
        results = model.predict(source=images_folder, conf=0.25, save=True)
        print("Inference completed. Results saved to 'runs/detect/predict'.")

        # Safer way to show results without crashing your screen
        print("\nPress any key to see the next image. Press 'q' to quit viewing.")
        for result in results:
            # result.plot() returns a numpy array (image) that OpenCV can read
            annotated_img = result.plot()
            cv2.imshow("Detection Result", annotated_img)

            # Wait for a key press. If 'q' is pressed, break the loop
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
