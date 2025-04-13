import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechToText:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained("phowhisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("phowhisper-base").to(self.device)

    def to_text(self, waveform, sample_rate):
        """
        Chuyển đổi tín hiệu âm thanh (waveform) thành văn bản.
        
        Args:
            waveform (torch.Tensor): Dãy các số trong khoảng [-1, 1] biểu diễn sóng âm thanh.
            sample_rate (int): Tần suất lấy mẫu của tín hiệu âm thanh (16000hz).
        
        Returns:
            str: Văn bản được sinh ra từ tín hiệu âm thanh.
        """
        input_features = self.processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)


        output_ids = self.model.generate(
            input_features.half(),  
            max_length=500,
            num_beams=5,
            length_penalty=0.8,
            temperature=0.7,
            early_stopping=True
        )[0]
        
        text = self.processor.decode(output_ids, skip_special_tokens=True)

        return text

