import gradio as gr
import cv2
import json
from datetime import timedelta
import os
from vlmagent import llavaImageCaptioner
from embed import CaptionIndexer

captioner = llavaImageCaptioner()
indexer = CaptionIndexer()

default_prompt = """You are an intelligent traffic monitoring assistant. Analyze this traffic scene image and identify:
1. Vehicle movements (e.g., turning left/right, going straight, waiting at signal).
2. Pedestrian activities (e.g., crossing the road, waiting at zebra crossing).
3. Traffic light status (red, yellow, green) and its position.
4. Any traffic rule violations, such as:
   - Vehicle running a red light
   - Pedestrian crossing on red signal
   - Vehicle blocking a pedestrian crossing
   - Illegal U-turns

Describe each finding clearly and mention the timestamp or frame number (if available). Your response should be structured as:
- **Timestamp/frame**: [e.g., 00:01:30]
  - **Vehicle Activity**: ...
  - **Pedestrian Activity**: ...
  - **Traffic Light**: ...
  - **Violations**: ...

Be concise and accurate."""

def seconds_to_timestamp(seconds):
    return str(timedelta(seconds=seconds))




####### to summarixe a video based on frames
def analyze_video(video, prompt):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    results = []

    output_dir = "gradio_frames"
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 60 == 0:  
            frame_path = os.path.join(output_dir, f"frame_{int(fps)*frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            caption = captioner.generate_caption(frame_path, prompt)
            timestamp = seconds_to_timestamp(frame_id // int(fps))

            results.append({
                "image_id": os.path.basename(frame_path),
                "caption": caption,
                "timestamp": timestamp,
                "source": os.path.basename(video)
            })

        frame_id += 1

    cap.release()

    with open("captions_output.json", "w") as f:
        json.dump(results, f, indent=2)

    indexer.build_index("captions_output.json")
    indexer.save_index()

    summary_text = "\n\n".join(
        [f" {r['timestamp']}: {r['caption']}" for r in results[:5]]
    )
    return "Analysis complete", summary_text


#Q and A  based on generated summary enhance answer by paasing image again

def query_summary(user_query):
    results = indexer.search(user_query, top_k=1)
    if not results:
        return "No relevant frame found."

    r = results[0]
    image_path = f"gradio_frames/{r['image_id']}"
    deeper_prompt = (
        f"This scene shows: {r['caption']}\n\n"
        "Please summarize this situation, identify safety concerns, and explain possible outcomes."
    )
    deeper_response = captioner.generate_caption(image_path, deeper_prompt)
    return f"[{r['timestamp']}] {deeper_response}"

with gr.Blocks() as demo:
    gr.Markdown("## Traffic Scene Violation Analyzer")

    with gr.Tab("Step 1: Upload Video"):
        video_input = gr.Video(label="📹 Upload a Traffic Scene Video")
        prompt_input = gr.Textbox(label="Instruction Prompt for VLM", value=default_prompt, lines=12)
        analyze_btn = gr.Button("Analyze Video")
        status = gr.Textbox(label="Status")
        summary_output = gr.Textbox(label="Summary", lines=10)

    with gr.Tab("Step 2: Ask a Question"):
        user_query = gr.Textbox(label="Ask a Question (e.g., Any violations?)")
        query_btn = gr.Button(" Get Answer")
        query_output = gr.Textbox(label=" Deeper Answer", lines=6)

    analyze_btn.click(analyze_video, inputs=[video_input, prompt_input], outputs=[status, summary_output])
    query_btn.click(query_summary, inputs=[user_query], outputs=[query_output])

demo.launch()





#code without gradio 


# import cv2
# import json
# from vlmagent import llavaImageCaptioner
# from datetime import timedelta
# from embed import CaptionIndexer



# prompt="""You are an intelligent traffic monitoring assistant. Analyze this traffic scene image and identify:

# 1. Vehicle movements (e.g., turning left/right, going straight, waiting at signal).
# 2. Pedestrian activities (e.g., crossing the road, waiting at zebra crossing).
# 3. Traffic light status (red, yellow, green) and its position.
# 4. Any traffic rule violations, such as:
#    - Vehicle running a red light
#    - Pedestrian crossing on red signal
#    - Vehicle blocking a pedestrian crossing
#    - Illegal U-turns

# Describe each finding clearly and mention the timestamp or frame number (if available). Your response should be structured as:

# - **Timestamp/frame**: [e.g., 00:01:30]
#   - **Vehicle Activity**: ...
#   - **Pedestrian Activity**: ...
#   - **Traffic Light**: ...
#   - **Violations**: ...

# Be concise and accurate.
# """
# def seconds_to_timestamp(seconds):
#     return str(timedelta(seconds=seconds))

# def generate_frame_captions(video_path, output_json_path):
#     indexer = CaptionIndexer()
#     captioner = llavaImaQwenVLImageCaptioner
#     cap = cv2.VideoCapture(video_path)
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_id = 0
#     results = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % 60 == 0:  # 1 frame per second
#             frame_path = f"/home/purushotham/Music/ATASK/frame_{int(fps)*frame_id}.jpg"
#             cv2.imwrite(frame_path, frame)



#             caption = captioner.generate_caption(frame_path, prompt)
#             # print(caption)
#             timestamp = seconds_to_timestamp(frame_id // int(fps))

#             results.append({
#                 "image_id": f"frame_{int(fps)*frame_id}.jpg",
#                 "caption": caption,
#                 "timestamp": timestamp,
#                 "source": video_path
#             })

#             # print(f"[{timestamp}] Frame {frame_id}: {caption}")

#         frame_id += 1

#     cap.release()

#     with open(output_json_path, "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"\n Captions saved to {output_json_path}")

    
#     indexer.build_index("captions_output.json")
    
#     indexer.save_index()

#     query = "Violation"
#     results = indexer.search(query, top_k=1)
#     # print("\n Top Results:")

#     # for r in results:
#     #     print(f"- [{r['timestamp']}] {r['caption']}")
    
#     for r in results:
#         image_path = f"/home/purushotham/Music/ATASK/{r['image_id']}"
#         deeper_prompt = (
#             f"This scene shows: {r['caption']}\n\n"
#             "Please summarize this situation, identify safety concerns, and explain possible outcomes."
#         )
#         deeper_response = captioner.generate_caption(image_path, deeper_prompt)
#         print(f"\n[{r['timestamp']}] Deeper Analysis:\n{deeper_response}")



# if __name__ == "__main__":
#     generate_frame_captions(
#         "/home/purushotham/Downloads/test_sample1.webm",
#         "captions_output.json"
#     )


