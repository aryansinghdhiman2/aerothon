import math

def calculate_gps_coordinates(initial_latitude, initial_longitude, altitude_and_sensor_height, focal_length, image_width, image_height, target_pixel_x, target_pixel_y, veh_heading):
    # Assuming heading angle is 0 (North-facing camera)
    heading_angle = veh_heading

    # Convert heading angle to radians
    heading_rad = math.radians(heading_angle)

    # Calculate Ground Sample Distance (GSD)
    gsd = (altitude_and_sensor_height) / (focal_length * image_height)

    # Convert pixel coordinates to meters
    displacement_x = (target_pixel_x - image_width / 2) * gsd
    displacement_y = (target_pixel_y - image_height / 2) * gsd

    # Adjust displacement based on heading angle
    adjusted_displacement_x = displacement_x * math.cos(heading_rad) - displacement_y * math.sin(heading_rad)
    adjusted_displacement_y = displacement_x * math.sin(heading_rad) + displacement_y * math.cos(heading_rad)

    # Translate meters to longitude and latitude
    earth_radius = 6371000  # Earth radius in meters
    latitude_change = adjusted_displacement_y / earth_radius
    longitude_change = adjusted_displacement_x / (earth_radius * math.cos(math.radians(initial_latitude)))

    new_latitude = initial_latitude + math.degrees(latitude_change)
    new_longitude = initial_longitude + math.degrees(longitude_change)

    return new_latitude, new_longitude
# Example usage

if(__name__=="__main__"):
    print(calculate_gps_coordinates(30.748021, 76.756599,15,3.04,640,480,320,240,180))
    pass