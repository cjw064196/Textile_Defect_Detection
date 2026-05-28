import ffmpeg


def convert_to_browser_playable(
    input_path: str = "test_media/output.mp4",
    output_path: str = "test_media/output_browser.mp4",
) -> None:
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            vcodec="libx264",
            acodec="aac",
            preset="medium",
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        .overwrite_output()
        .run()
    )


if __name__ == "__main__":
    convert_to_browser_playable()

