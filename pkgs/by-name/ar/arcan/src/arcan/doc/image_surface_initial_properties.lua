-- image_surface_initial_properties
-- @short: Retrieve a table describing the initial storage state for the specified object.
-- @inargs: vid
-- @outargs: proptbl
-- @longdescr: This function aliases image_surface_initial and is kept around
-- for legacy reasons. Please see ref:image_surface_initial instead.
-- @group: image
-- @cfunction: getimageinitprop
-- @related: image_surface_resolve, image_surface_properties
function main()
#ifdef MAIN
	a = load_image("test.png");
	resize_image(a, 32, 32);
	iprop = image_surface_initial_properties(a);
	cprop = image_surface_properties(a);

	print(string.format("initial_w: %d, inital_h: %d, current_w: %d, current_h: %d",
		iprop.width, iprop.height, cprop.width, cprop.height));
#endif
end
