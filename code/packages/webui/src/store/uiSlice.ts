import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

interface UiState {
  selectedAttachmentId: string | null;
}

const initialState: UiState = {
  selectedAttachmentId: null,
};

const uiSlice = createSlice({
  name: "ui",
  initialState,
  reducers: {
    openAttachment(state, action: PayloadAction<string>) {
      state.selectedAttachmentId = action.payload;
    },
    closeAttachment(state) {
      state.selectedAttachmentId = null;
    },
  },
});

export const { openAttachment, closeAttachment } = uiSlice.actions;
export default uiSlice.reducer;
