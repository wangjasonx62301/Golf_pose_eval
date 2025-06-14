import cv2, json, re, os


def slow_video(video, slow_factor=2):
    cap = cv2.VideoCapture(video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_fps = fps / slow_factor

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"slow_{video}", fourcc, new_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

    cap.release()
    out.release()

def merge_videos_with_advice(skeleton_video, ball_video, output_video, json_file, font_color=(255, 255, 255), slow_factor=6):

    skeleton_cap = cv2.VideoCapture(skeleton_video)
    ball_cap = cv2.VideoCapture(ball_video)

    frame_width = int(skeleton_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(skeleton_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(skeleton_cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    line_type = cv2.LINE_AA

    with open(json_file, "r") as f:
        advice_data = json.load(f)

    for person in advice_data["frames"]:
        for advice in person["persons"]:
            frame_idx = person["frame"]
            skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ball_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            skeleton_ret, skeleton_frame = skeleton_cap.read()
            ball_ret, ball_frame = ball_cap.read()
            if not (skeleton_ret and ball_ret):
                continue

            frame = cv2.addWeighted(skeleton_frame, 1, ball_frame, 1, 0)

            y_offset = 0
            for _, v in advice.get("advice", {}).items():
                match = re.search(r"Correction:\s*(.*?)(?=\s*<\|endoftext\|>)", v)
                if match:
                    text = match.group(1).strip()
                    position = (50, frame_height - 100 + y_offset)
                    cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness, line_type)
                    y_offset += 40

            out.write(frame)

            frame_folder = f"frame/with_text/{os.path.splitext(ball_video)[0]}"
            os.makedirs(frame_folder, exist_ok=True)
            cv2.imwrite(f"{frame_folder}/{frame_idx}.png", frame)

    skeleton_cap.release()
    ball_cap.release()
    out.release()

    print(f"Merged video saved to: {output_video}")

    if slow_factor > 1:
        slow_video(output_video, slow_factor)


def main(
    skeleton_video, ball_video, output_video, json_file, font_color=(255, 255, 255)
):
    skeleton_cap = cv2.VideoCapture(skeleton_video)
    ball_cap = cv2.VideoCapture(ball_video)

    frame_width = int(skeleton_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(skeleton_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(skeleton_cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    line_type = cv2.LINE_AA

    with open(json_file, "r") as f:
        advice = json.load(f)

    # advices = {}

    for person in advice["frames"]:
        for advice in person["persons"]:
            skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, person["frame"])
            ball_cap.set(cv2.CAP_PROP_POS_FRAMES, person["frame"])
            # advices[person["frame"]] = []

            skeleton_ret, skeleton_frame = skeleton_cap.read()
            ball_ret, ball_frame = ball_cap.read()

            # frame = np.hstack((skeleton_frame, ball_frame))
            frame = cv2.addWeighted(skeleton_frame, 1, ball_frame, 1, 0)

            text = ""
            if len(advice["advice"]):
                y = 0
                for k, v in advice["advice"].items():
                    match = re.search(r"Correction:\s*(.*?)(?=\s*<\|endoftext\|>)", v)
                    if match:
                        text = match.group(1).strip()

                    # advices[person["frame"]].append(text)
                    # print(text)
                    position = (50, frame_height - 100 + y)
                    cv2.putText(
                        frame,
                        text,
                        position,
                        font,
                        font_scale,
                        font_color,
                        font_thickness,
                        line_type,
                    )

                    y += 40

            out.write(frame)
            path = f"frame/with_text/{ball_video.replace('.mp4', '')}"
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(f"{path}/{person['frame']}.png", frame)
            # print(advice["advice"])

    skeleton_cap.release()
    ball_cap.release()
    out.release()
    # print(advices)


if __name__ == "__main__":
    main(
        "overlay_result.mp4",
        "504.mp4",
        "result.mp4",
        "/home/public_datasets/golf/keypoint_combined_json/keypoints_504-1_aligned_combined.json",
    )
    slow_video("result.mp4", 6)
    
    # cap = cv2.VideoCapture("504.mp4")
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    # cnt = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     cv2.imwrite(f"{cnt}.png", frame)