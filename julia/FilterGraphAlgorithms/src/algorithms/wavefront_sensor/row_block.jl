"""Transport-neutral identity and ordering state for one detector row frame."""
mutable struct RowBlockFrame
    active::Bool
    sequence::UInt64
    metadata::SampleMetadata
    expected_row::Int
    pending_discontinuity::Bool
    current_discontinuity::Bool
end

RowBlockFrame() = RowBlockFrame(false, 0, _EMPTY_SAMPLE_METADATA, 0, false, false)

function _start_row_block_frame!(frame::RowBlockFrame, sample::SampleMetadata)
    frame.active = true
    frame.sequence = sample.sequence
    frame.metadata=sample
    frame.expected_row = 0
    frame.current_discontinuity =
        frame.pending_discontinuity || sample.discontinuity
    frame.pending_discontinuity = false
    return frame
end

function _abandon_row_block_frame!(frame::RowBlockFrame)
    frame.active = false
    frame.expected_row = 0
    frame.pending_discontinuity = true
    frame.current_discontinuity = false
    return frame
end

function _complete_row_block_frame!(frame::RowBlockFrame)
    frame.active = false
    frame.expected_row = 0
    frame.pending_discontinuity = false
    frame.current_discontinuity = false
    return frame
end

function _reset_row_block_frame!(frame::RowBlockFrame)
    frame.active = false
    frame.sequence = 0
    frame.expected_row = 0
    frame.pending_discontinuity = false
    frame.current_discontinuity = false
    return frame
end

function _accept_row_block!(
    frame::RowBlockFrame,
    sample::SampleMetadata,
    first_row::Int,
    block_rows::Int,
    image_rows::Int,
)
    if sample.corrupted
        _abandon_row_block_frame!(frame)
        return false, 0
    end

    if iszero(first_row)
        superseded = frame.active
        superseded && _abandon_row_block_frame!(frame)
        _start_row_block_frame!(frame, sample)
        superseded && (frame.current_discontinuity = true)
    elseif !frame.active ||
           !_same_row_block_metadata(frame.metadata,sample) ||
           sample.discontinuity
        _abandon_row_block_frame!(frame)
        return false, 0
    end

    if first_row != frame.expected_row || first_row > image_rows - block_rows
        _abandon_row_block_frame!(frame)
        return false, 0
    end
    end_row = first_row + block_rows
    if sample.terminal != (end_row == image_rows)
        _abandon_row_block_frame!(frame)
        return false, 0
    end
    frame.expected_row = end_row
    return true, end_row
end
