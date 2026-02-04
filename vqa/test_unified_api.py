import requests
import os

# CONFIG
URL = "http://aih.cse.iitd.ac.in:8000"
IMAGE_PATH = "sample2.png"  # <--- CHANGE THIS to your actual image path
OBJECT_TO_FIND = "swimming pool"         # <--- Object to find for BBox test
QUESTION = "What color is the running track surface?"

def run_test(name, endpoint, data=None):
    print(f"\n--- Testing {name} ({endpoint}) ---")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File {IMAGE_PATH} not found.")
        return

    # Prepare the multipart upload
    files = {'image': open(IMAGE_PATH, 'rb')}
    
    try:
        response = requests.post(f"{URL}{endpoint}", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Success!")
            import json
            # Pretty print the JSON response
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        files['image'].close()

if __name__ == "__main__":
    # 1. Test Caption (VLM Only)
    import time
    # s = time.time()
    # run_test("Captioning", "/caption")
    # # run_test("Routing", "/router")

    # # 2. Test BBox (SAM3 Only)
    # run_test("BBox Detection", "/bbox", {"object_name": OBJECT_TO_FIND})

    # 3. Test VQA (Full Pipeline)
    # 1. ATTRIBUTE
    # print("--- Running Attribute Test ---")
    # start = time.time()
    # run_test("VQA (Attribute)", "/vqa/attribute", {
    #     "question": "What color is the surface of the running track?",
    #     "use_sam": "true",
    #     "classes": "running track" 
    # })
    # print(f"Attribute Test completed in: {time.time() - start:.4f} seconds\n")

    # 2. FILTERING
    print("--- Running Filtering Test ---")
    start = time.time()
    run_test("VQA (Filtering)", "/vqa/filtering", {
        "question": "Locate all swimming pools.",
        "use_sam": "true",
        "classes": "swimming pool" 
    })
    print(f"Filtering Test completed in: {time.time() - start:.4f} seconds\n")

    # # 3. BINARY
    # print("--- Running Binary Test ---")
    # start = time.time()
    # run_test("VQA (Binary)", "/vqa/binary", {
    #     "question": "Is there a swimming pool visible near the baseball field?",
    #     "use_sam": "true",
    #     "classes": "swimming pool|baseball field" 
    # })
    # print(f"Binary Test completed in: {time.time() - start:.4f} seconds\n")

    # # 4. NUMERICAL
    # print("--- Running Numerical Test ---")
    # start = time.time()
    # run_test("VQA (Numerical)", "/vqa/numerical", {
    #     "question": "What is the area of the blue region in the larger swimming pool in meters square?",
    #     "use_sam": "true",
    #     "classes": "swimming pool" 
    # })
    # print(f"Numerical Test completed in: {time.time() - start:.4f} seconds\n")