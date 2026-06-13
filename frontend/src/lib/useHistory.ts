import { useCallback, useReducer } from 'react'

interface HState<T> {
  past: T[]
  present: T
  future: T[]
}

type HAction<T> =
  | { type: 'set'; next: T }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'reset'; value: T }

function reducer<T>(s: HState<T>, a: HAction<T>): HState<T> {
  switch (a.type) {
    case 'set':
      if (a.next === s.present) return s
      return { past: [...s.past, s.present], present: a.next, future: [] }
    case 'undo':
      if (!s.past.length) return s
      return {
        past: s.past.slice(0, -1),
        present: s.past[s.past.length - 1],
        future: [s.present, ...s.future],
      }
    case 'redo':
      if (!s.future.length) return s
      return { past: [...s.past, s.present], present: s.future[0], future: s.future.slice(1) }
    case 'reset':
      return { past: [], present: a.value, future: [] }
  }
}

/** Undo/redo history over an immutable value (used for the editable layers). */
export function useHistory<T>(initial: T) {
  const [state, dispatch] = useReducer(reducer<T>, { past: [], present: initial, future: [] })
  return {
    present: state.present,
    set: useCallback((next: T) => dispatch({ type: 'set', next }), []),
    undo: useCallback(() => dispatch({ type: 'undo' }), []),
    redo: useCallback(() => dispatch({ type: 'redo' }), []),
    reset: useCallback((value: T) => dispatch({ type: 'reset', value }), []),
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
  }
}
