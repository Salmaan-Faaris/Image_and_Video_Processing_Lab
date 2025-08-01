import cv2
import numpy as np
import argparse
import os

def box_average(img, k):
    if k <= 0:
        return img.copy()
    return cv2.blur(img, (k, k))

def highpass(img, blurred):
    # High-frequency component estimate
    return cv2.subtract(img.astype(np.float32), blurred.astype(np.float32))

def edge_strength(gray):
    # Sobel-based edge magnitude sum
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    return np.sum(np.abs(mag))

def compute_score(orig, filtered):
    # Convert to float32 for metrics
    orig = orig.astype(np.float32)
    filtered = filtered.astype(np.float32)

    # Estimate noise via highpass std deviation (original minus its own blur with small kernel)
    # Use a small blur as baseline for "natural" low-frequency
    baseline_blur = box_average(orig, 3)
    orig_high = highpass(orig, baseline_blur)
    filtered_high = highpass(filtered, box_average(filtered, 3))

    noise_orig = np.std(orig_high)
    noise_filt = np.std(filtered_high)
    noise_reduction = noise_orig - noise_filt  # higher is better

    # Edge preservation: compare edge strength
    orig_gray = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    filt_gray = cv2.cvtColor(filtered.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edge_orig = edge_strength(orig_gray)
    edge_filt = edge_strength(filt_gray)
    # Relative edge loss
    edge_loss = max(0.0, (edge_orig - edge_filt) / (edge_orig + 1e-8))  # lower is better

    # Combined: reward noise reduction, penalize edge loss
    score = noise_reduction / (1 + edge_loss)
    return {
        "noise_orig": noise_orig,
        "noise_filt": noise_filt,
        "noise_reduction": noise_reduction,
        "edge_orig": edge_orig,
        "edge_filt": edge_filt,
        "edge_loss": edge_loss,
        "combined_score": score,
    }

def visualize_and_save(out_dir, original, results, best_key):
    os.makedirs(out_dir, exist_ok=True)
    # Save original
    cv2.imwrite(os.path.join(out_dir, "original.png"), original)
    print(f"Saved original -> {os.path.join(out_dir, 'original.png')}")

    # Stack comparison strip: original + each filtered
    h, w = original.shape[:2]
    margin = 10
    labels = []
    canvases = []
    for key, img in results.items():
        # Annotate with label
        canvas = img.copy()
        cv2.putText(canvas, key, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if key == best_key:
            # highlight best with green border
            cv2.rectangle(canvas, (0, 0), (w-1, h-1), (0, 255, 0), 4)
        canvases.append(canvas)
        labels.append(key)
        cv2.imwrite(os.path.join(out_dir, f"{key}.png"), img)
        print(f"Saved {key} -> {os.path.join(out_dir, f'{key}.png')}")

    # Compose a grid (original + filtered)
    grid = [original] + [results[k] for k in results]
    combined = np.hstack([cv2.resize(x, (w, h)) for x in grid])
    cv2.imwrite(os.path.join(out_dir, "comparison_strip.png"), combined)
    print(f"Saved comparison strip -> {os.path.join(out_dir, 'comparison_strip.png')}")

def main():
    parser = argparse.ArgumentParser(description="Denoise a noisy image by box averaging with various kernel sizes and pick best tradeoff.")
    parser.add_argument("image", help="Path to noisy input image")
    parser.add_argument("--kernels", nargs="+", type=int, default=[2, 8, 16, 32, 128],
                        help="List of averaging kernel sizes to try")
    parser.add_argument("--output-dir", "-o", default="denoise_results", help="Directory to save outputs")
    parser.add_argument("--show", "-s", action="store_true", help="Display result summary windows")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    results = {}
    scores = {}
    for k in args.kernels:
        key = f"avg_{k}x{k}"
        filtered = box_average(img, k)
        results[key] = filtered
        scores[key] = compute_score(img, filtered)

    # Determine best by highest combined_score
    best_key = max(scores.items(), key=lambda kv: kv[1]["combined_score"])[0]
    best_info = scores[best_key]

    # Print a summary table
    print("\n=== Summary ===")
    for key, metrics in scores.items():
        print(f"{key:12} | noise_red: {metrics['noise_reduction']:7.2f} | edge_loss: {metrics['edge_loss']:.3f} | combined_score: {metrics['combined_score']:.2f}")

    print(f"\n=> Best kernel/filter: {best_key} (highest combined score)")

    # Save visuals
    visualize_and_save(args.output_dir, img, results, best_key)

    # Optionally show
    if args.show:
        cv2.imshow("Original", img)
        for key, imgf in results.items():
            winname = f"{key} - score {scores[key]['combined_score']:.2f}"
            cv2.imshow(winname, imgf)
        print("Press any key in image windows to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
