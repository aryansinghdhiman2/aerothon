import math

image_height = 480
image_width = 640
# Converted to meters
focal_length = 3.04 / 1000 
sensor_width = 3.68 / 1000
sensor_height = 2.76 / 1000

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

def calculate_body_ned_coordinates(altitude,heading,x,y):
    gsd_y = (altitude * sensor_height) / ( focal_length * image_height )
    gsd_x = (altitude * sensor_width) / ( focal_length * image_width )

    # print(gsd_x,gsd_y)

    gsd = min(gsd_x,gsd_y) / 2

    x_meters = gsd * x
    y_meters = gsd * y

    # print(x_meters,y_meters)

    heading_rads = math.radians(heading)

    x_ned = x_meters * math.cos(heading_rads) + y_meters * math.sin(heading_rads)
    y_ned = -x_meters * math.sin(heading_rads) + y_meters * math.cos(heading_rads)

    return -y_ned,x_ned

def calculate_body_frd_coordinates(altitude,heading,x,y):
    gsd_y = (altitude * sensor_height) / ( focal_length * image_height )
    gsd_x = (altitude * sensor_width) / ( focal_length * image_width )

    # print(gsd_x,gsd_y)

    gsd = min(gsd_x,gsd_y) / 2

    x_meters = gsd * x
    y_meters = gsd * y
    
    return y_meters,x_meters

if(__name__=="__main__"):
    # print(calculate_gps_coordinates(30.7481128, 76.7565859,15,3.04 / 1000,640,480,365,196,70))
    # print(calculate_body_ned_coordinates(15,90,0,240))
    print([i for i in calculate_body_frd_coordinates(15,90,0,240)])
    pass