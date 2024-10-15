import os

# 라벨 파일이 있는 디렉토리 경로
label_directory = 'datasets/face/valid/labels'

# 디렉토리 내 모든 파일을 순회
for filename in os.listdir(label_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(label_directory, filename)
        
        # 파일을 읽고 클래스 번호를 1로 변경
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        with open(file_path, 'w') as file:
            for line in lines:
                parts = line.strip().split()
                parts[0] = '1'  # 클래스 번호를 1로 변경
                file.write(' '.join(parts) + '\n')

print("클래스 번호를 1로 변경하는 작업이 완료되었습니다.")
