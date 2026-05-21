"""
QR 码生成器 —— 使用 qrcode 库生成 PNG 二维码
"""
import io


def generate_qr_png(text: str, size: int = 240, border: int = 2) -> bytes:
    """生成 QR 码 PNG 图像字节。"""
    import qrcode
    qr = qrcode.QRCode(
        version=None,  # 自动选择版本
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=border,
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
