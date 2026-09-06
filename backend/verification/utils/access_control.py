"""
Role-Based Access Control for Workflow Execution - Task 7.2.3

Provides RBAC to control who can execute workflows, view results, and modify
configurations:
- WorkflowPermission enum — granular permissions
- WorkflowRole enum — predefined roles
- AccessControlError — structured exception
- RolePermissions — mapping of roles to allowed permissions
- AccessController — permission checking with decorator support

Default role-permission mappings:
- SALES_REP:     EXECUTE_WORKFLOW, VIEW_RESULTS
- SALES_MANAGER: EXECUTE_WORKFLOW, VIEW_RESULTS, ESCALATE, VIEW_AUDIT_LOG
- ADMIN:         all permissions
- SYSTEM:        all permissions

Requirements:
- 7.2.3: Access control for workflow execution with role-based permissions
"""

from __future__ import annotations

import functools
import logging
from enum import Enum
from typing import Callable, Dict, FrozenSet, Set

logger = logging.getLogger("backend.verification.access_control")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowPermission(str, Enum):
    """Granular permissions for workflow operations."""

    EXECUTE_WORKFLOW = "execute_workflow"
    VIEW_RESULTS = "view_results"
    MODIFY_CONFIG = "modify_config"
    ESCALATE = "escalate"
    VIEW_AUDIT_LOG = "view_audit_log"


class WorkflowRole(str, Enum):
    """Predefined roles for workflow access control."""

    SALES_REP = "sales_rep"
    SALES_MANAGER = "sales_manager"
    ADMIN = "admin"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class AccessControlError(Exception):
    """
    Raised when a role lacks the required permission.

    Attributes:
        role: The role that was denied.
        permission: The permission that was required.
    """

    def __init__(self, role: WorkflowRole, permission: WorkflowPermission) -> None:
        self.role = role
        self.permission = permission
        super().__init__(
            f"AccessControlError: role '{role.value}' does not have permission '{permission.value}'"
        )


# ---------------------------------------------------------------------------
# Role → Permission mapping
# ---------------------------------------------------------------------------

# All permissions as a frozen set for convenience
_ALL_PERMISSIONS: FrozenSet[WorkflowPermission] = frozenset(WorkflowPermission)

#: Default mapping of roles to their allowed permissions.
RolePermissions: Dict[WorkflowRole, FrozenSet[WorkflowPermission]] = {
    WorkflowRole.SALES_REP: frozenset(
        {
            WorkflowPermission.EXECUTE_WORKFLOW,
            WorkflowPermission.VIEW_RESULTS,
        }
    ),
    WorkflowRole.SALES_MANAGER: frozenset(
        {
            WorkflowPermission.EXECUTE_WORKFLOW,
            WorkflowPermission.VIEW_RESULTS,
            WorkflowPermission.ESCALATE,
            WorkflowPermission.VIEW_AUDIT_LOG,
        }
    ),
    WorkflowRole.ADMIN: _ALL_PERMISSIONS,
    WorkflowRole.SYSTEM: _ALL_PERMISSIONS,
}


# ---------------------------------------------------------------------------
# AccessController
# ---------------------------------------------------------------------------


class AccessController:
    """
    Role-based access controller for workflow operations.

    Uses a configurable role-permission mapping (defaults to
    :data:`RolePermissions`).  All permission checks are logged for
    auditability.

    Usage::

        controller = AccessController()

        # Raises AccessControlError if denied
        controller.check_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)

        # Returns bool
        if controller.has_permission(WorkflowRole.ADMIN, WorkflowPermission.MODIFY_CONFIG):
            ...

        # Decorator
        @controller.require_permission(WorkflowRole.SALES_MANAGER, WorkflowPermission.ESCALATE)
        def escalate_workflow(state):
            ...
    """

    def __init__(
        self,
        role_permissions: Dict[WorkflowRole, FrozenSet[WorkflowPermission]] | None = None,
    ) -> None:
        """
        Initialise the AccessController.

        Args:
            role_permissions: Custom role-permission mapping.  Defaults to the
                module-level :data:`RolePermissions` dict.
        """
        self._role_permissions: Dict[WorkflowRole, FrozenSet[WorkflowPermission]] = (
            role_permissions if role_permissions is not None else dict(RolePermissions)
        )

    # ------------------------------------------------------------------
    # Core permission API
    # ------------------------------------------------------------------

    def check_permission(
        self, role: WorkflowRole, permission: WorkflowPermission
    ) -> None:
        """
        Assert that *role* has *permission*.

        Args:
            role: The role to check.
            permission: The required permission.

        Raises:
            AccessControlError: If the role does not have the permission.
        """
        if not self.has_permission(role, permission):
            logger.warning(
                "Access denied: role='%s' permission='%s'",
                role.value,
                permission.value,
            )
            raise AccessControlError(role=role, permission=permission)

        logger.debug(
            "Access granted: role='%s' permission='%s'",
            role.value,
            permission.value,
        )

    def has_permission(
        self, role: WorkflowRole, permission: WorkflowPermission
    ) -> bool:
        """
        Return True if *role* has *permission*, False otherwise.

        Args:
            role: The role to check.
            permission: The permission to test.

        Returns:
            bool
        """
        allowed = self._role_permissions.get(role, frozenset())
        return permission in allowed

    def get_permissions(self, role: WorkflowRole) -> Set[WorkflowPermission]:
        """
        Return the set of permissions granted to *role*.

        Args:
            role: The role whose permissions to retrieve.

        Returns:
            A (mutable copy of the) set of :class:`WorkflowPermission` values.
        """
        return set(self._role_permissions.get(role, frozenset()))

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def require_permission(
        self, role: WorkflowRole, permission: WorkflowPermission
    ) -> Callable:
        """
        Decorator that guards a function with a permission check.

        The check is performed at call time using the *role* and *permission*
        provided when the decorator is applied.

        Args:
            role: The role that must have the permission.
            permission: The required permission.

        Returns:
            A decorator that wraps the target function.

        Raises:
            AccessControlError: When the decorated function is called and the
                role does not have the required permission.

        Example::

            @controller.require_permission(WorkflowRole.ADMIN, WorkflowPermission.MODIFY_CONFIG)
            def update_config(new_config):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.check_permission(role, permission)
                return func(*args, **kwargs)

            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_controller: AccessController | None = None


def get_access_controller() -> AccessController:
    """
    Return the module-level singleton :class:`AccessController`.

    Creates the instance on first call using the default
    :data:`RolePermissions` mapping.
    """
    global _default_controller
    if _default_controller is None:
        _default_controller = AccessController()
    return _default_controller


def reset_access_controller() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_controller
    _default_controller = None


__all__ = [
    "WorkflowPermission",
    "WorkflowRole",
    "AccessControlError",
    "RolePermissions",
    "AccessController",
    "get_access_controller",
    "reset_access_controller",
]
