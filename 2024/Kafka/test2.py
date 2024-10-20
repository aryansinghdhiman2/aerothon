import dronekit_sitl

sitl = dronekit_sitl.start_default()
print(sitl.connection_string())