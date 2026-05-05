import pygame
import serial
import time
import argparse
import numpy as np

BAUD_RATE = 921600
WIDTH = 320
HEIGHT = 240
FRAME_PREAMBLE = b"===FRAME===\n"
PREVIEW_W = 160
PREVIEW_H = 120

def capture_and_display_loop(port: str):
    print(f"Opening serial port {port}... ", end="")
    try:
        serial_port = serial.Serial(port, BAUD_RATE, timeout=3.0)
        serial_port.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"Failed to open port {port}: {exc}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smart Lock Live Feed")

    print("Connection established. Starting Live Feed & AI Monitor...\n")
    serial_port.write(b'S')
    time.sleep(0.5)
    serial_port.reset_input_buffer()
    print("Stream started, waiting for first frame...")

    running = True
    last_frame_rgb = None

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d and last_frame_rgb is not None:
                        from PIL import Image
                        img = Image.fromarray(last_frame_rgb)
                        img.save("debug_esp32_frame.jpg", quality=95)
                        print("[DEBUG] Saved debug_esp32_frame.jpg")

            chunk = serial_port.read_until(FRAME_PREAMBLE)
            if not chunk.endswith(FRAME_PREAMBLE):
                continue

            ai_text = chunk[:-len(FRAME_PREAMBLE)].decode('utf-8', errors='ignore').strip()
            if ai_text:
                print(ai_text)

            frame_rgb565 = serial_port.read(PREVIEW_W * PREVIEW_H * 2)
            if len(frame_rgb565) != PREVIEW_W * PREVIEW_H * 2:
                serial_port.reset_input_buffer()
                continue

            raw = np.frombuffer(frame_rgb565, dtype=np.uint8).reshape(PREVIEW_H, PREVIEW_W, 2)
            byte1 = raw[:, :, 0].astype(np.uint16)
            byte2 = raw[:, :, 1].astype(np.uint16)
            r = (byte1 & 0xF8).astype(np.uint8)
            g = (((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3)).astype(np.uint8)
            b = ((byte2 & 0x1F) << 3).astype(np.uint8)
            last_frame_rgb = np.stack([r, g, b], axis=2)

            small_surface = pygame.surfarray.make_surface(last_frame_rgb.swapaxes(0, 1))
            surface = pygame.transform.scale(small_surface, (WIDTH, HEIGHT))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

    finally:
        serial_port.close()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM5", help="Serial port")
    args = parser.parse_args()
    capture_and_display_loop(args.port)