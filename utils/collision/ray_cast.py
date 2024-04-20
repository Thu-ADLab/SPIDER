def ray_cast_check(point, polygon):
    # Check if the point is inside the polygon using the ray-casting algorithm
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

if __name__== "__main__":
    polygon = [(0, 0), (0, 5), (5, 5), (5, 0)]
    point = (2, 2)
    is_inside = ray_cast_check(point, polygon)
    print(is_inside)