import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math, numpy as np, cv2, threading, time,socket

running = True
S = 0.0001

port=4545
udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
udp.bind(("127.0.0.1",port))
alpha,beta=0,0

def handle_udp():
    global alpha,beta,running
    while running:
        time.sleep(0.01)
        data,addr=udp.recvfrom(1024)
        data=data.decode()
        alpha,beta=map(float,data.split(","))
        alpha *=S
        beta *=S
        print(f"Received: alpha={alpha}, beta={beta}")
    udp.close()
    
threading.Thread(target=handle_udp,daemon=True).start()

# --- Video capture setup (reader thread will only read frames) ---
vid = cv2.VideoCapture("hand.mp4")
if not vid.isOpened():
    raise RuntimeError("Cannot open video file mm.mp4")

frame_lock = threading.Lock()
frame_rgb = None        # shared frame in RGB+flipped format ready for GL upload
new_frame_available = False
w,h=100,100

def video_reader_loop():
    global frame_rgb, new_frame_available, running
    while running:
        ret, frame = vid.read()
        if not ret:
            # loop video
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = vid.read()
            if not ret:
                time.sleep(0.01)
                continue
        # Convert BGR->RGB and flip vertically (OpenGL expects bottom-left origin)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        with frame_lock:
            frame_rgb = frame.copy()
            new_frame_available = True
        # Sleep a little to avoid hogging CPU (reader faster than display)
        time.sleep(1/30)

# start reader thread
reader_thread = threading.Thread(target=video_reader_loop, daemon=True)
reader_thread.start()

# --- Pygame / OpenGL setup ---
pygame.init()
DISPLAY = (800, 600)
pygame.display.set_mode(DISPLAY, OPENGL | DOUBLEBUF)
clock = pygame.time.Clock()

glEnable(GL_DEPTH_TEST)
glClearColor(0.0, 0.0, 0.0, 1.0)
glMatrixMode(GL_PROJECTION)
gluPerspective(45, DISPLAY[0] / DISPLAY[1], 0.1, 100.0)
glMatrixMode(GL_MODELVIEW)

# --- Camera ---
camPos = np.array([0.0, 0.0, 3.0], np.float64)
camR   = np.array([1.0, 0.0, 0.0], np.float64)
camU   = np.array([0.0, 1.0, 0.0], np.float64)
camF   = np.array([0.0, 0.0, -1.0], np.float64)

def load_texture_from_file(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot load {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 0)  # flip vertically for OpenGL
    h, w, _ = img.shape

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, img)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_id, w, h

def draw_background(tex_id):
    glDisable(GL_DEPTH_TEST)  # background ignores depth
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)  # simple 2D projection

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(1, 0)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(0, 1)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

    # restore matrices
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)



def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def rotate_vec(v, axis, angle):
    axis = normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (v * cos_a +
            np.cross(axis, v) * sin_a +
            axis * np.dot(axis, v) * (1 - cos_a))

# --- Create single texture object (no image yet) ---
texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glBindTexture(GL_TEXTURE_2D, 0)

bg_tex_id, bg_w, bg_h = load_texture_from_file("image.png")


def draw_triangle():
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0); glVertex3f(-0.5, -0.5, 0.0)
    glColor3f(0.0, 1.0, 0.0); glVertex3f( 0.5, -0.5, 0.0)
    glColor3f(0.0, 0.0, 1.0); glVertex3f( 0.0,  0.5, 0.0)
    glEnd()

def draw_textured_quad(tex_id, w=100, h=100, x=0, y=0, z=-2.0):
    """Draw a textured quad at depth z, preserving aspect ratio of w:h."""
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    aspect = w / h
    half_w = aspect
    half_h = 1.0

    glBegin(GL_QUADS)
    glColor3f(1, 1, 1)
    glTexCoord2f(0, 0); glVertex3f(x-half_w, y-half_h, z)
    glTexCoord2f(1, 0); glVertex3f(x+half_w, y-half_h, z)
    glTexCoord2f(1, 1); glVertex3f(x+half_w, y+half_h, z)
    glTexCoord2f(0, 1); glVertex3f(x-half_w,  y+half_h, z)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

px, py = pygame.mouse.get_pos()

# --- Main loop: upload frames to GPU from main thread only ---
try:
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                raise KeyboardInterrupt()


        camF = rotate_vec(camF, camU, -beta)
        camR = rotate_vec(camR, camU, -beta)
        camF = rotate_vec(camF, camR, -alpha)
        camU = rotate_vec(camU, camR, -alpha)
        
        beta,alpha=0,0    
        
        # move
        keys = pygame.key.get_pressed()
        v = 0.02
        camPos += v*camF

        # If a new frame is available, upload it to the single texture (main thread)
        if new_frame_available:
            with frame_lock:
                frame = frame_rgb.copy()
                # mark consumed (clear flag)
                new_frame_available = False

            h, w, _ = frame.shape
            glBindTexture(GL_TEXTURE_2D, texture_id)
            # First-time or if size changed, use glTexImage2D to allocate; otherwise glTexSubImage2D is faster.
            # We'll just use glTexImage2D here for simplicity (works robustly).
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, frame)
            glBindTexture(GL_TEXTURE_2D, 0)

        # render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        draw_background(bg_tex_id)
        
        glLoadIdentity()
        gluLookAt(*camPos, *(camPos + camF), *camU)

        # draw_triangle()
        draw_textured_quad(texture_id, w, h, -2.5, 0 ,-5)

        pygame.display.flip()
        clock.tick(25)

except KeyboardInterrupt:
    # cleanup
    running = False
    vid.release()
    pygame.quit()
