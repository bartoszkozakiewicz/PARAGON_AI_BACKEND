from PIL import Image
from paddleocr import PaddleOCR
# Initialize the PaddleOCR reader
ocr = PaddleOCR(use_gpu=True)


path = "/content/drive/MyDrive/img_kref_scan/k_scan0.jpg"
path_wrap = "/content/drive/MyDrive/img_kerf_warp_persp/k_wrap0.jpg"

# Open an image file
img = Image.open(path)
img_wrap = Image.open(path_wrap)

# Perform text detection and recognition
paddle_results = ocr.ocr(path)
print(paddle_results[0][0])
for idx,result in enumerate(paddle_results[0]):
  # print(idx,"BBOX: ",result[0], "TEXT: ",result[1])
  bbox = result[0][0] + result[0][2]
  paddle_results.append({"x0":int(bbox[0]),"y0":int(bbox[1]),"x2":int(bbox[2]),"y2":int(bbox[3]) , "word": result[1]})
  print("WORD:",result[1],"\n")
# pd.DataFrame(prepared_results)