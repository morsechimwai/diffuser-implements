import argparse
from sdxl_txt2img import txt2img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--guidance_scale", type=float, default=7)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompt = f"{args.prompt}, 4k"
    negative = "ugly, deformed, blurry, low quality, lowres, bad anatomy"

    img = txt2img(
        prompt=prompt,
        negative_prompt=negative,
    )

    img.save(f"{args.output}.png")
    print(f"saved: {args.output}.png")

if __name__ == "__main__":
    main()
