-- snapshot_target
-- @short: Request that a frameserver serializes its internal state into the
-- specified resource.
-- @inargs: vid:target, string:resource
-- @inargs: vid:target, string:resource, number:namespace, string:description
-- @outargs: bool:status
-- @longdescr:
-- Create a new *resource* in the APPL_STATE_RESOURCE namespace and
-- send a file reference there to *target* requesting that it stores
-- its state from the resource. This returns *false* if the file
-- could not be created due to permissions, and *true* if the file
-- was created and sent successfully.
-- If *namespace* is provided and the namespace is writable, the
-- resource will instead be created within that namespace. Depending
-- if the type of the segment is a CLIPBOARD or a normal application
-- window with a bchunk state exposed or in a mouse-drag, the client
-- will interpret the operation as a save, paste or drag-and-drop.
-- If a *description* is provided, a short (~60ch) text type descriptor
-- of the data can be amended. The convention is to treat this as a
-- simplified MIME- type, with the application/ prefix being the default
-- (and can thus be omitted).
-- @note: Normally only the APPL_TEMP_RESOURCE is writable, but
-- the mechanism is reserved for user- defined namespaces as well.
-- @group: targetcontrol
-- @cfunction: targetsnapshot
