import cv2
import os

# penamaan file video
def get_name_video(folder, prefix="video_gerakan", ext=".mp4"):
  files = os.listdir(folder)
  number = []
  
  for f in files:
    if f.startswith(prefix) and f.endswith(ext):
      try:
        num = int(f.replace(prefix, "-", "").replace(ext, ""))
        number.append(num)
      except:
        pass
  next_num = max(number)+1 if number else 1
  return os.path.join(folder, f"{prefix}_{next_num}{ext}")
# inisiasi kamera
cap = cv2.VideoCapture(0)
# output directory
output_dir = './data/videos'
os.makedirs(output_dir, exist_ok=True)
path_video = get_name_video(output_dir)
# mendefinisikan codec
eks_video = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(path_video, eks_video, 20.0, (640, 480))

print("Instruksi:")
print("tekan s untuk menyimpan vidoe")
print("merekam video, tekan q untuk berhenti")

while (cap.isOpened()):
  ret, frame = cap.read()
  if ret:
    # tulis frame kedalam file input
    output_video.write(frame)
    #tampilkan video
    cv2.imshow('merekam video', frame)
    key = cv2.waitKey(1)
    # menyimpan video
    if key & 0xFF == ord('s'):
      print(f"Video disimpan di: {path_video}")
      break
    # mengakhiri rekaman
    if key & 0xFF == ord('q'):
        print("Rekaman dihentikan.")
        break
  else:
    break
# 3. Lepaskan sumber daya
cap.release()
output_video.release()
cv2.destroyAllWindows()