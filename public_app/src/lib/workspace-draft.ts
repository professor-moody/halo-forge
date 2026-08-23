import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, ApiError, type WorkspaceDraft } from "@/lib/api";

type WorkspaceDraftOptions<T> = {
  surface: string;
  draftKey: string;
  name: string;
  value: T;
  enabled?: boolean;
  onRestore: (value: T) => void;
};

/**
 * Keeps long workstation forms recoverable without making local browser
 * storage the source of truth. Existing drafts are always reviewed before
 * the current form can replace them.
 */
export function useWorkspaceDraft<T>({
  surface,
  draftKey,
  name,
  value,
  enabled = true,
  onRestore,
}: WorkspaceDraftOptions<T>) {
  const [candidate, setCandidate] = useState<WorkspaceDraft<T> | null>(null);
  const [resolved, setResolved] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const valueRef = useRef(value);
  valueRef.current = value;

  const query = useQuery({
    queryKey: ["workspace-draft", surface, draftKey],
    queryFn: () => api.workspaceDraft<T>(surface, draftKey),
    enabled,
    retry: false,
    staleTime: Infinity,
  });

  useEffect(() => {
    if (query.data) {
      setCandidate(query.data);
      setResolved(false);
      return;
    }
    if (query.isError) {
      // 404 is the normal first-use response. Other failures keep the form
      // usable; the save indicator truthfully reports that persistence is off.
      setResolved(true);
    }
  }, [query.data, query.isError]);

  const save = useMutation({
    mutationFn: (next: T) => api.saveWorkspaceDraft(surface, draftKey, { name, content: next }),
    onSuccess: (draft) => setSavedAt(draft.updated_at ?? new Date().toISOString()),
  });
  const remove = useMutation({
    mutationFn: () => api.deleteWorkspaceDraft(surface, draftKey),
    onSuccess: () => {
      setCandidate(null);
      setResolved(true);
      setSavedAt(null);
    },
  });

  useEffect(() => {
    if (!enabled || !resolved || query.isLoading) return;
    const timer = window.setTimeout(() => save.mutate(valueRef.current), 900);
    return () => window.clearTimeout(timer);
    // `save` is intentionally omitted: React Query returns a new mutation
    // wrapper while its stable mutate function performs the same operation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, name, query.isLoading, resolved, surface, draftKey, value]);

  function restore() {
    if (!candidate) return;
    onRestore(candidate.content);
    setCandidate(null);
    setResolved(true);
  }

  function discard() {
    remove.mutate();
  }

  return {
    candidate,
    restore,
    discard,
    clear: () => api.deleteWorkspaceDraft(surface, draftKey),
    isLoading: query.isLoading,
    isSaving: save.isPending,
    savedAt,
    saveError: save.error instanceof Error ? save.error.message : null,
    unavailable: query.error instanceof ApiError && query.error.status !== 404,
  };
}
