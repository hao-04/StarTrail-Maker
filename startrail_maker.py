import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QMessageBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

class VideoProcessorThread(QThread):
    # 定义信号，用于与主线程通信
    progress_signal = pyqtSignal(int, int)  # 当前帧, 总帧数
    preview_signal = pyqtSignal(QImage)     # 预览图像
    finished_signal = pyqtSignal(str)       # 完成提示信息
    error_signal = pyqtSignal(str)          # 错误提示信息

    def __init__(self, input_path, output_path, resolution_scale=1.0, 
                 decay_factor=1.0, brightness_threshold=0, confirm_frames=3):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.resolution_scale = resolution_scale
        self.decay_factor = decay_factor
        self.brightness_threshold = brightness_threshold
        self.confirm_frames = confirm_frames
        self.is_cancelled = False

    def run(self):
        try:
            # 1. 打开输入视频
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.error_signal.emit("无法打开输入视频文件，请检查文件格式。")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 计算缩放后的分辨率
            new_width = int(width * self.resolution_scale)
            new_height = int(height * self.resolution_scale)

            # 2. 配置输出视频写入器 (使用 mp4v 编码)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (new_width, new_height))

            # 3. 读取第一帧
            ret, first_frame = cap.read()
            if not ret:
                self.error_signal.emit("无法读取视频帧。")
                cap.release()
                return

            if self.resolution_scale != 1.0:
                first_frame = cv2.resize(first_frame, (new_width, new_height))

            # 使用 float32 进行计算，避免衰减时的精度损失
            first_f = first_frame.astype(np.float32)

            # 初始化轨迹层（仅记录星轨，不含背景）
            trail_layer = np.zeros((new_height, new_width, 3), dtype=np.float32)
            # 持续性计数器：追踪每个像素连续为亮的帧数
            # 只有连续多帧都亮的像素（星星）才被确认加入轨迹
            # 飞机/卫星在同一像素位置仅停留1-2帧，无法通过确认
            persistence = np.zeros((new_height, new_width), dtype=np.float32)
            # 持续性衰减系数：像素不再亮时，计数器快速衰减
            persist_decay = 0.5

            # 处理第一帧
            if self.brightness_threshold > 0:
                bright_mask = np.max(first_f, axis=2) > self.brightness_threshold
            else:
                bright_mask = np.ones((new_height, new_width), dtype=bool)
            persistence[bright_mask] += 1.0

            # 第一帧：如果不需要多帧确认，直接加入轨迹
            if self.confirm_frames <= 1:
                trail_layer = np.where(bright_mask[:, :, np.newaxis], first_f, 0)

            output = np.maximum(first_f, trail_layer)
            out.write(np.clip(output, 0, 255).astype(np.uint8))
            self.progress_signal.emit(1, total_frames)

            current_frame_idx = 1
            # 计算预览更新间隔（每秒更新2次预览，避免UI刷新过快卡顿）
            preview_interval = max(1, int(fps / 2))

            # 4. 逐帧处理
            while True:
                if self.is_cancelled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if self.resolution_scale != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height))

                frame_f = frame.astype(np.float32)

                # ====== 核心算法：持续性确认 + 衰减 + 合成 ======

                # Step 1: 提取当前帧的亮像素候选
                if self.brightness_threshold > 0:
                    brightness = np.max(frame_f, axis=2)
                    bright_mask = brightness > self.brightness_threshold
                else:
                    bright_mask = np.ones((new_height, new_width), dtype=bool)

                # Step 2: 更新持续性计数器
                # 亮像素：计数 +1（上限防溢出）
                # 非亮像素：计数快速衰减（每帧 ×0.5）
                persistence[bright_mask] = np.minimum(
                    persistence[bright_mask] + 1.0,
                    self.confirm_frames * 3.0
                )
                persistence[~bright_mask] *= persist_decay

                # Step 3: 衰减轨迹层，让旧轨迹逐渐变暗
                trail_layer *= self.decay_factor

                # Step 4: 只有通过多帧确认的像素才加入轨迹
                # 星星在同一位置持续多帧 → 计数器达到阈值 → 确认为星星
                # 飞机/卫星仅停留1-2帧 → 计数器不足 → 被过滤
                confirmed = persistence >= self.confirm_frames
                if np.any(confirmed):
                    confirmed_3ch = confirmed[:, :, np.newaxis]
                    trail_layer = np.where(
                        confirmed_3ch,
                        np.maximum(trail_layer, frame_f),
                        trail_layer
                    )

                # Step 5: 合成输出 = 当前帧 + 轨迹层 (取较亮值)
                # 飞机/卫星虽未入轨迹层，但仍出现在当前帧中，形成自然残影
                output = np.maximum(frame_f, trail_layer)
                output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
                out.write(output_uint8)

                current_frame_idx += 1

                # 更新进度条 (每 5 帧更新一次 UI，提升性能)
                if current_frame_idx % 5 == 0 or current_frame_idx == total_frames:
                    self.progress_signal.emit(current_frame_idx, total_frames)

                # 更新预览图
                if current_frame_idx % preview_interval == 0:
                    rgb_image = cv2.cvtColor(output_uint8, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.preview_signal.emit(q_img.copy())

            # 5. 释放资源
            cap.release()
            out.release()

            if self.is_cancelled:
                self.finished_signal.emit("处理已取消，已保存部分视频。")
            else:
                self.finished_signal.emit("处理完成！星轨视频已成功保存。")

        except Exception as e:
            self.error_signal.emit(f"处理过程中发生异常: {str(e)}")


class StarTrailMaker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StarTrail Maker - 星轨视频生成器")
        self.resize(800, 800)
        self.init_ui()
        self.processor_thread = None

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- 1. 文件选择区域 ---
        file_layout = QHBoxLayout()
        self.btn_select_input = QPushButton("选择输入视频")
        self.btn_select_input.setMinimumHeight(40)
        self.btn_select_input.clicked.connect(self.select_input)
        self.lbl_input = QLabel("未选择文件")
        self.lbl_input.setStyleSheet("color: gray;")
        file_layout.addWidget(self.btn_select_input)
        file_layout.addWidget(self.lbl_input, 1)
        layout.addLayout(file_layout)

        output_layout = QHBoxLayout()
        self.btn_select_output = QPushButton("选择导出路径")
        self.btn_select_output.setMinimumHeight(40)
        self.btn_select_output.clicked.connect(self.select_output)
        self.lbl_output = QLabel("未选择路径")
        self.lbl_output.setStyleSheet("color: gray;")
        output_layout.addWidget(self.btn_select_output)
        output_layout.addWidget(self.lbl_output, 1)
        layout.addLayout(output_layout)

        # --- 2. 参数设置区域 ---
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("视频分辨率缩放:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["100% (原画质)", "75%", "50%", "25%"])
        self.combo_scale.setToolTip("降低分辨率可减少内存占用并提升处理速度")
        settings_layout.addWidget(self.combo_scale)
        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # 轨迹衰减速度
        decay_layout = QHBoxLayout()
        decay_layout.addWidget(QLabel("轨迹衰减速度:"))
        self.combo_decay = QComboBox()
        self.combo_decay.addItems([
            "无衰减（经典全量叠加）",
            "极慢衰减（超长拖尾）",
            "慢速衰减（长拖尾）",
            "中速衰减（推荐）",
            "快速衰减（短拖尾）"
        ])
        self.combo_decay.setCurrentIndex(3)
        self.combo_decay.setToolTip(
            "控制旧轨迹变暗的速度。\n"
            "• 无衰减：所有亮像素永久保留（原始算法）\n"
            "• 有衰减：旧轨迹逐帧变暗，瞬态杂物会自动消失"
        )
        decay_layout.addWidget(self.combo_decay)
        decay_layout.addStretch()
        layout.addLayout(decay_layout)

        # 亮度过滤阈值
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("亮度过滤阈值:"))
        self.combo_threshold = QComboBox()
        self.combo_threshold.addItems([
            "无过滤 (0)",
            "极低 (15)",
            "低 (30) — 推荐夜景",
            "中 (60)",
            "高 (100)",
            "极高 (150)"
        ])
        self.combo_threshold.setCurrentIndex(2)
        self.combo_threshold.setToolTip(
            "只有亮度超过此阈值的像素才会被加入轨迹。\n"
            "• 较低阈值：保留更多细节，但可能包含少量噪点\n"
            "• 较高阈值：只保留最亮的星星，有效过滤地面杂光和噪点"
        )
        threshold_layout.addWidget(self.combo_threshold)
        threshold_layout.addStretch()
        layout.addLayout(threshold_layout)

        # 瞬态物体过滤（飞机/卫星）
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("瞬态物体过滤:"))
        self.combo_confirm = QComboBox()
        self.combo_confirm.addItems([
            "关闭（不过滤）",
            "轻度（2帧确认）",
            "标准（3帧确认）— 推荐",
            "强力（5帧确认）",
            "极强（8帧确认）"
        ])
        self.combo_confirm.setCurrentIndex(2)
        self.combo_confirm.setToolTip(
            "过滤飞机、卫星等快速移动的瞬态亮点。\n"
            "原理：只有连续多帧出现在同一像素位置的亮点才会被确认为星星并加入轨迹。\n"
            "飞机/卫星在每个像素位置仅停留1-2帧，无法通过确认，自然被过滤。\n"
            "• 关闭：所有亮像素直接加入轨迹（适用于延时摄影）\n"
            "• 标准：需连续3帧确认，适合大多数实时录制视频\n"
            "• 极强：需连续8帧确认，可能过滤部分暗星"
        )
        filter_layout.addWidget(self.combo_confirm)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # --- 3. 预览区域 ---
        self.lbl_preview = QLabel("实时预览区域")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #1e1e1e; color: #888888; border-radius: 8px;")
        self.lbl_preview.setMinimumHeight(400)
        layout.addWidget(self.lbl_preview, 1)

        # --- 4. 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # --- 5. 控制按钮 ---
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_start.clicked.connect(self.start_processing)
        
        self.btn_cancel = QPushButton("取消 / 停止")
        self.btn_cancel.setMinimumHeight(50)
        self.btn_cancel.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 14px;")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_cancel)
        layout.addLayout(control_layout)

        self.input_path = ""
        self.output_path = ""

    def select_input(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_name:
            self.input_path = file_name
            self.lbl_input.setText(file_name)

    def select_output(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "保存视频文件", "startrail_output.mp4", "MP4 Video (*.mp4)")
        if file_name:
            self.output_path = file_name
            self.lbl_output.setText(file_name)

    def start_processing(self):
        if not self.input_path or not self.output_path:
            QMessageBox.warning(self, "警告", "请先选择输入视频和导出路径！")
            return

        # 解析缩放比例
        scale_str = self.combo_scale.currentText()
        scale = 1.0
        if "75%" in scale_str: scale = 0.75
        elif "50%" in scale_str: scale = 0.5
        elif "25%" in scale_str: scale = 0.25

        # 解析轨迹衰减系数
        decay_map = {0: 1.0, 1: 0.999, 2: 0.997, 3: 0.995, 4: 0.99}
        decay = decay_map.get(self.combo_decay.currentIndex(), 0.995)

        # 解析亮度过滤阈值
        threshold_map = {0: 0, 1: 15, 2: 30, 3: 60, 4: 100, 5: 150}
        threshold = threshold_map.get(self.combo_threshold.currentIndex(), 30)

        # 解析瞬态物体过滤帧数
        confirm_map = {0: 1, 1: 2, 2: 3, 3: 5, 4: 8}
        confirm = confirm_map.get(self.combo_confirm.currentIndex(), 3)

        # 更新 UI 状态
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_preview.setText("准备处理...")

        # 启动工作线程
        self.processor_thread = VideoProcessorThread(
            self.input_path, self.output_path, scale, decay, threshold, confirm
        )
        self.processor_thread.progress_signal.connect(self.update_progress)
        self.processor_thread.preview_signal.connect(self.update_preview)
        self.processor_thread.finished_signal.connect(self.processing_finished)
        self.processor_thread.error_signal.connect(self.processing_error)
        self.processor_thread.start()

    def cancel_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.is_cancelled = True
            self.btn_cancel.setEnabled(False)
            self.lbl_preview.setText("正在停止并保存文件...")

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"处理中: %p% ({current}/{total} 帧)")

    def update_preview(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        # 保持比例缩放适应预览区域
        scaled_pixmap = pixmap.scaled(self.lbl_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(scaled_pixmap)

    def processing_finished(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.information(self, "提示", msg)

    def processing_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "错误", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局样式，使其更贴近 Windows 11 风格
    app.setStyle("Fusion") 
    window = StarTrailMaker()
    window.show()
    sys.exit(app.exec())