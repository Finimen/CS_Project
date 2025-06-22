# People Counter System

![image](https://github.com/user-attachments/assets/908a5e78-847a-4cb7-83d7-e78eaa178fde)

A simple computer vision-based system for counting people crossing a virtual line in real-time video streams.

## 📌 Overview

This Python script uses **OpenCV** and **Haar cascades** to detect and track people's faces, counting entries and exits based on their movement relative to a configurable virtual line.

## 🛠 Team

- **Сапсай Константин**
- **Карпенко Данил**
- **Павлов Артем**

## 🔧 Key Features

- **Real-time face detection** using Haar cascade classifier
- **Object tracking** with KCF tracker
- **Virtual line crossing detection** with configurable Y-position
- **Automatic logging** of events (entries/exits) with timestamps
- **Visual feedback** with bounding boxes and counter display
- **Data export** to CSV and visualization

## 🛠 Technical Details

```python
# Core components:
- Haar Cascade face detection
- KCF object tracking
- Pandas for data logging
- Matplotlib for visualization
