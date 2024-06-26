-- target_input
-- @short: Forward input data to a frameserver.
-- @alias: input_target
-- @inargs: tgtvid, inputtbl
-- @outargs: resbool
-- @longdescr:  This function takes a properly formated *inputtbl* and
-- repackages it into an event structure that is added to the event queue
-- of the frameserver specified by *tgtvid*. The function returns true
-- if and only if the event was properly formatted and could be successfully
-- added to the incoming event queue of the frameserver.
--
-- A properly formatted table has the following fields:
--
-- universal:(devid, subid, kind) optional:(pts, gesture, label).
-- The *devid* and *subid* fields are used to distinguish between different
-- devices and subaxes or buttons. The *kind* field indicates the type of the
-- input and determines how the table will be interpreted. The optional
-- *pts* field includes an optional implementation defined relative 64-bit
-- timestamp in milisecond resolution. The optional *gesture* is a
-- boolean flag that indiciates if the event is a result from abstract
-- gesture analysis or not, and alters the client interpretation of *label'.
--
-- If gesture is set, the contents of *label* should match one of the
-- following: 'touch', 'dbltouch', 'click', 'dblclick', 'drag', 'swipe',
-- 'fling', 'press', 'release', 'drag', 'drop', 'openpinch', 'closepinch',
-- 'rotate'.
-- If *gesture* is not set/true, the contents of *label* should match a
-- previously frameserver- provided labelhint, or, in the context of
-- a game, the more abstract 'PLAYERn_BUTTONm', 'PLAYERn_AXISm'.
--
-- If *kind* is set to *touch* the required additional fields are:
-- pressure (0..1, linear scale), size (px), x (px) and y (px)
--
-- If *kind* is set to *analog* the required additional fields are:
-- relative (true/_false_), samples (indexed table of up to 4 int16_t
-- ranged values). These values are all the same for generic analog
-- sources, such as game device axes.
--
-- If *kind* is set to *analog* the possible additional fields are:
-- *mouse* (true/_false_), indicate if the samples come from a cursor-
-- type device or not. If so, the interpretation of sample values is
-- more complicated as there is a large variability in how underlying
-- device layers provide their samples. If *relative* is set to true,
-- the order is (*rel_x*, *abs_x*,  *rel_y*, *abs_y*), otherwise the
-- order is (*abs_x*, *rel_x*, *rel_y*, *abs_y*). The reason for this is
-- that the relative indicator shows the primary origin of the samples,
-- while the other is a result of some state estimator and can therefore
-- be less accurate. All permutations of relative, absolute, two samples
-- or four samples need to be accounted for.
--
-- If *kind* is set to *digital*, the required additional fields are:
-- active (true/_false_), if the button is in a pressed state or not
-- and the possible additional fields are:
-- *translated* (true/_false_) and if translated is set, the required
-- additional fields are:
-- *keysym* (value should match SDL1.2 table of symnames), *modifiers*
-- (16 bit bitfield), *utf8* (single unicode character as utf-8)
-- *number* (undescript field, typically used to carry on device or
-- OS specific code).
--
-- If *kind* is set to *eyes*, the required additional fields are:
-- blink_left, blink_right (true/false) if the eye lids are closed or not
-- gaze_x1, gaze_y1 (float) screen coordinates for first gaze point
-- gaze_x2, gaze_y2 (float) screen coordinates for second gaze point
-- present (true/false) if the user is by the screen or not
-- head_x, head_y, head_z (float) head position relative to the tracker
-- head_rx, head_ry, head_rz (float) euler-angle for head
--
-- @note: instead of preparing these tables manually, it's recommended to
-- inplace modify those you get from the applname_input event handler
-- @note: arguments are flexible, table argument will be used as input table,
-- number argument will be used as frameserver vid reference.
-- @note: the kind field can also be supplied as a bool to cut down
-- on costly string compares by using the kind type (analog, digital, touch)
-- as boolean key set to true, and the same applies to (translated, mouse).
-- @note: mouse events does not necessarily translate into an event on the
-- event-queue, the engine may chose to use a memory mapped interface. This
-- depends on the devid (0 for accelerated cursor).
-- @related: message_target
-- @group: targetcontrol
-- @cfunction: targetinput

