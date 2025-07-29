import os
import math


def extend_polyline_to_edges(points, image_width, image_height):
    if len(points) < 2:
        clipped_single_point = []
        if points:
            x, y = points[0]
            clipped_x = max(0.0, min(x, float(image_width - 1)))
            clipped_y = max(0.0, min(y, float(image_height - 1)))
            clipped_single_point.append((clipped_x, clipped_y))
        return clipped_single_point

    x_min, y_min = 0.0, 0.0
    x_max, y_max = float(image_width - 1), float(image_height - 1)
    epsilon = 1e-6

    def get_intersection(p_ref, dx_dir, dy_dir, edge_type, edge_val):
        if edge_type == 'x':
            if abs(dx_dir) < epsilon:
                return None
            t = (edge_val - p_ref[0]) / dx_dir
            y_intersect = p_ref[1] + t * dy_dir
            if y_min - epsilon <= y_intersect <= y_max + epsilon:
                return (edge_val, y_intersect, t)
        elif edge_type == 'y':
            if abs(dy_dir) < epsilon:
                return None
            t = (edge_val - p_ref[1]) / dy_dir
            x_intersect = p_ref[0] + t * dx_dir
            if x_min - epsilon <= x_intersect <= x_max + epsilon:
                return (x_intersect, edge_val, t)
        return None

    p_start_polyline = points[0]
    p_next_polyline = points[1]

    dx_start = p_next_polyline[0] - p_start_polyline[0]
    dy_start = p_next_polyline[1] - p_start_polyline[1]

    candidate_starts = []

    res = get_intersection(p_start_polyline, dx_start, dy_start, 'x', x_min)
    if res and res[2] < -epsilon:
        candidate_starts.append((res[0], res[1], res[2]))

    res = get_intersection(p_start_polyline, dx_start, dy_start, 'y', y_min)
    if res and res[2] < -epsilon:
        candidate_starts.append((res[0], res[1], res[2]))

    res = get_intersection(p_start_polyline, dx_start, dy_start, 'x', x_max)
    if res and res[2] < -epsilon:
        candidate_starts.append((res[0], res[1], res[2]))

    res = get_intersection(p_start_polyline, dx_start, dy_start, 'y', y_max)
    if res and res[2] < -epsilon:
        candidate_starts.append((res[0], res[1], res[2]))

    extended_start_point = p_start_polyline
    if candidate_starts:
        candidate_starts.sort(key=lambda x: x[2], reverse=True)
        extended_start_point = (candidate_starts[0][0], candidate_starts[0][1])

    p_end_polyline = points[-1]
    p_prev_polyline = points[-2]

    dx_end = p_end_polyline[0] - p_prev_polyline[0]
    dy_end = p_end_polyline[1] - p_prev_polyline[1]

    candidate_ends = []

    res = get_intersection(p_end_polyline, dx_end, dy_end, 'x', x_min)
    if res and res[2] > epsilon:
        candidate_ends.append((res[0], res[1], res[2]))

    res = get_intersection(p_end_polyline, dx_end, dy_end, 'y', y_min)
    if res and res[2] > epsilon:
        candidate_ends.append((res[0], res[1], res[2]))

    res = get_intersection(p_end_polyline, dx_end, dy_end, 'x', x_max)
    if res and res[2] > epsilon:
        candidate_ends.append((res[0], res[1], res[2]))

    res = get_intersection(p_end_polyline, dx_end, dy_end, 'y', y_max)
    if res and res[2] > epsilon:
        candidate_ends.append((res[0], res[1], res[2]))

    extended_end_point = p_end_polyline
    if candidate_ends:
        candidate_ends.sort(key=lambda x: x[2])
        extended_end_point = (candidate_ends[0][0], candidate_ends[0][1])

    final_polyline = []

    if math.hypot(extended_start_point[0] - p_start_polyline[0], extended_start_point[1] - p_start_polyline[1]) > epsilon:
        final_polyline.append(extended_start_point)

    final_polyline.extend(points)

    if math.hypot(extended_end_point[0] - p_end_polyline[0], extended_end_point[1] - p_end_polyline[1]) > epsilon:
        final_polyline.append(extended_end_point)

    final_clipped_and_unique_points = []
    seen = set()
    for x, y in final_polyline:
        clipped_x = max(x_min, min(x, x_max))
        clipped_y = max(y_min, min(y, y_max))

        p_str = f"{clipped_x:.2f},{clipped_y:.2f}"
        if p_str not in seen:
            final_clipped_and_unique_points.append((clipped_x, clipped_y))
            seen.add(p_str)

    return final_clipped_and_unique_points


def process_and_extend_single_line(data_string, image_width, image_height):
    values = list(map(float, data_string.split()))

    if len(values) % 2 != 0:
        values.pop()

    points = [(values[i], values[i+1]) for i in range(0, len(values), 2)]

    extended_points = extend_polyline_to_edges(
        points, image_width, image_height)

    formatted_points = []
    for p in extended_points:
        formatted_points.append(f"{p[0]:.2f} {p[1]:.2f}")

    return " ".join(formatted_points)


if __name__ == "__main__":
    imageWidth = 1200
    imageHeight = 590
    imgCount = 0

    baseDir = 'frame_culane_backup/driver_00_01frame'
    if not os.path.exists(baseDir):
        print(
            f"Warning: Base directory '{baseDir}' does not exist. No files will be processed.")

    while True:
        lineLabelFilename = f'{imgCount:05}.lines.txt'
        lineLabelPath = os.path.join(baseDir, lineLabelFilename)

        if not os.path.exists(lineLabelPath):
            print(f"No more files found, stopped at: {lineLabelPath}")
            break

        try:
            with open(lineLabelPath, 'r') as f:
                lines = f.readlines()

            processedFileContent = []
            for originalLine in lines:
                singleExtendedLine = process_and_extend_single_line(
                    originalLine.strip(), imageWidth, imageHeight)
                if singleExtendedLine:
                    processedFileContent.append(singleExtendedLine)

            finalLinesToWrite = processedFileContent[:20]

            with open(lineLabelPath, 'w') as f:
                if finalLinesToWrite:
                    f.write("\n".join(finalLinesToWrite) + "\n")
                else:
                    f.write("")

            print(
                f"Processed and extended lines for: {lineLabelPath}. Total lines written: {len(finalLinesToWrite)}")

        except FileNotFoundError:
            print(
                f"Error: The file '{lineLabelPath}' was not found after initial check.")
        except Exception as e:
            print(f"An error occurred while processing {lineLabelPath}: {e}")

        imgCount += 1
